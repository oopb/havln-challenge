"""Three canonical model roles, bounded HTTP transport and one-object JSON protocol."""
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
import re
from pathlib import Path
import socket
import signal
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from jsonschema import Draft7Validator, ValidationError

from scripts.export_cma_submission import atomic_json

PROMPT_VERSION = 'havln-v28-grounded-navigation-feedback'
ROLES = ('navigation_agent', 'visual_perception')


def object_schema(properties):
    return {'type': 'object', 'properties': properties,
            'required': list(properties), 'additionalProperties': False}


TEXT = {'type': 'string', 'maxLength': 1000}
FLAG = {'type': 'boolean'}


class APIError(RuntimeError):
    """A sanitized failure that is safe to put in navigation logs."""

    def __init__(self, message, category=None):
        super().__init__(message)
        self.category = category


class BudgetExceeded(APIError):
    pass


class RequestDeadlineExceeded(RuntimeError):
    pass


class RequestDeadlineUnavailable(APIError):
    pass


@contextmanager
def request_deadline(seconds):
    """Bound HTTP wall time in the Linux worker's main thread, including reads."""
    if (threading.current_thread() is not threading.main_thread() or
            not hasattr(signal, 'setitimer') or any(signal.getitimer(signal.ITIMER_REAL))):
        raise RequestDeadlineUnavailable('request_deadline_unavailable')
    previous = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise RequestDeadlineExceeded('total_request_deadline_exceeded')

    try:
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def usage_counts(usage):
    counts = {key: value for key, value in (usage or {}).items() if type(value) is int}
    reasoning = ((usage or {}).get('completion_tokens_details') or {}).get('reasoning_tokens')
    if type(reasoning) is int:
        counts['reasoning_tokens'] = reasoning
    cached = ((usage or {}).get('prompt_tokens_details') or {}).get('cached_tokens')
    if type(cached) is int:
        counts['cached_prompt_tokens'] = cached
        if type(counts.get('prompt_tokens')) is int and counts['prompt_tokens'] >= cached:
            counts['uncached_prompt_tokens_reported'] = counts['prompt_tokens'] - cached
            counts['uncached_prompt_tokens_reported_requests'] = 1
        elif type(counts.get('prompt_tokens')) is int:
            counts['cached_tokens_exceed_prompt_records'] = 1
    return counts


def specific_schema_error(error):
    """Resolve tagged union errors to the actual field, rather than a huge oneOf."""
    if not error.context:
        return error
    children = list(error.context)
    if error.validator == 'oneOf' and isinstance(error.instance, dict):
        for tag in ('action', 'tool', 'coordinate_space'):
            if tag not in error.instance:
                continue
            indices = [i for i, branch in enumerate(error.validator_value)
                       if branch.get('properties', {}).get(tag, {}).get('const') == error.instance[tag]]
            if indices:
                children = [e for e in children if e.schema_path[0] in indices]
                break
        else:
            # Undeclared image units are checked against the normalized branch.
            if any('coordinate_space' in b.get('properties', {}) for b in error.validator_value):
                children = [e for e in children if e.schema_path[0] == 0]
    leaves = [specific_schema_error(child) for child in children]
    return max(leaves or [error], key=lambda e: (e.validator not in ('const', 'type'), len(e.absolute_path)))


def provider_schema(schema):
    """Local repair policy is not part of a provider's JSON Schema dialect."""
    if isinstance(schema, list):
        return [provider_schema(v) for v in schema]
    if isinstance(schema, dict):
        return {k: provider_schema(v) for k, v in schema.items() if not k.startswith('x-')}
    return schema


def reject_json_constant(value):
    raise json.JSONDecodeError('Non-finite values are not valid JSON', value, 0)


def decode_response(content, allow_identical_duplicates=False):
    """Decode one JSON response without silently resolving conflicts.

    Only explicitly opted-in, read-only VLM observation responses may
    normalize duplicate keys with EXACTLY identical JSON values. Navigation
    tool decisions, conflicting duplicates and malformed structures remain
    strict errors. All normalized results still undergo schema validation.
    """
    normalization = None
    identical_duplicate_keys = []
    if isinstance(content, str):
        blocks = list(re.finditer(r'^\s*```([^\n]*)\n(.*?)^\s*```\s*$', content,
                                  re.MULTILINE | re.DOTALL))
        if len(blocks) == 1 and blocks[0].group(1).strip().lower() in ('json', ''):
            block = blocks[0]
            outside = content[:block.start()] + content[block.end():]
            whole = not outside.strip()
            if whole or block.group(1).strip().lower() == 'json':
                for start in re.finditer(r'[\[{]', outside):
                    try:
                        extra, _ = json.JSONDecoder().raw_decode(outside[start.start():])
                    except json.JSONDecodeError:
                        continue
                    if isinstance(extra, (dict, list)):
                        raise json.JSONDecodeError('Ambiguous JSON outside explicit block', content, 0)
                content = block.group(2)
                normalization = 'markdown_json_fence' if whole else 'unique_json_block'
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                # bool and int compare equal in Python but are different JSON
                # values; use both type identity and canonical serialization.
                if (not allow_identical_duplicates
                        or type(result[key]) is not type(value)
                        or canonical(result[key]) != canonical(value)):
                    label=('Conflicting duplicate JSON key: ' if
                           allow_identical_duplicates else
                           'Duplicate JSON key: ')
                    raise json.JSONDecodeError(label + key, '', 0)
                identical_duplicate_keys.append(key)
                continue
            result[key] = value
        return result
    decoder = json.JSONDecoder(
        parse_constant=reject_json_constant, object_pairs_hook=unique_pairs)
    def with_identical_duplicate_note(value, existing):
        if identical_duplicate_keys:
            names=','.join(sorted(set(identical_duplicate_keys)))
            note='identical_duplicate_keys['+names+']'
            return value, (existing+'+'+note if existing else note)
        return value, existing
    try:
        return with_identical_duplicate_note(decoder.decode(content),
                                             normalization)
    except json.JSONDecodeError as original_error:
        # Some Chat Completions providers leak a plain-language preface even
        # with response_format=json_object. Only recover ONE complete,
        # unambiguous JSON container; all existing schema/role validation
        # remains mandatory BEFORE any tool is dispatched.
        if not isinstance(content, str):
            raise
        candidates = []
        # A failed first parse must not contribute duplicate-key metadata to
        # a later, independently decoded embedded JSON candidate.
        identical_duplicate_keys.clear()
        consumed_until = 0
        for start, char in enumerate(content):
            if start < consumed_until or char not in '{[':
                continue
            try:
                value, length = decoder.raw_decode(content, start)
            except json.JSONDecodeError:
                continue
            if not isinstance(value, (dict, list)):
                continue
            candidates.append((start, length, value))
            consumed_until = length   # do not treat nested args as second JSON
            if len(candidates) > 1:
                break
        if len(candidates) != 1:
            raise original_error
        start, end, value = candidates[0]
        # Braces/brackets outside the sole container can represent another
        # action or an incomplete JSON fragment. Reject rather than guess.
        outside = content[:start] + content[end:]
        if any(c in outside for c in '{}[]'):
            raise original_error
        if not (content[:start].strip() or content[end:].strip()):
            raise original_error
        return with_identical_duplicate_note(value,
                                             'unique_embedded_json_container')


REASONING_LEVELS = frozenset(('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'))


def reasoning_payload(effort, mode):
    """Map the canonical effort to one supported Chat Completions extension."""
    if mode == 'reasoning_effort':
        return {'reasoning_effort': effort}
    if mode == 'thinking':
        # Vendor-specific boolean switch: cannot preserve precise effort.
        return {'thinking': {'type': 'disabled' if effort == 'none' else 'enabled'}}
    if mode == 'omit':
        return {}
    raise ValueError('Unknown reasoning wire mode')


def unsupported_reasoning_parameter(detail, field):
    """Only negotiate on explicit unsupported FIELD errors, never bad values."""
    try:
        data = json.loads(detail)
        error = data.get('error', data) if isinstance(data, dict) else {}
        if not isinstance(error, dict):
            error = {}
    except (ValueError, TypeError):
        error = {}
    message = str(error.get('message', detail)).lower()
    code = str(error.get('code', '')).lower()
    param = str(error.get('param') or '').lower()
    if param and param != field:
        return False
    if field not in message and param != field:
        return False
    if re.search(r'\b(invalid|unsupported|unknown|unrecognized)\s+(value|effort|level)\b', message):
        return False
    return (code in ('unknown_parameter', 'unsupported_parameter', 'extra_forbidden')
            or bool(re.search(
                r'\b(unknown parameter|unrecognized parameter|unsupported parameter|'
                r'not supported|not allowed|unexpected parameter|extra fields not permitted|'
                r'unexpected keyword|unknown field)\b', message)))


def unsupported_chat_parameter(detail, field):
    """True only for a clearly rejected wire field, never a generic HTTP 400.

    The standard unknown/unsupported-parameter check also covers rejection of
    max_tokens and max_completion_tokens. Special cases are narrowly scoped to
    model-specific temperature defaults and unsupported native JSON modes.
    """
    if unsupported_reasoning_parameter(detail, field):
        return True
    try:
        data = json.loads(detail)
        error = data.get('error', data) if isinstance(data, dict) else {}
        if not isinstance(error, dict):
            return False
    except (ValueError, TypeError):
        return False
    param = str(error.get('param') or '').lower()
    code = str(error.get('code') or '').lower()
    message = str(error.get('message') or '').lower()
    if param and param != field:
        return False
    if field not in message and param != field:
        return False
    if field == 'temperature':
        return ('default' in message and
                ('only' in message or 'must be' in message) and
                ('unsupported' in message or 'not supported' in message
                 or code in ('unsupported_value', 'invalid_value')))
    if field == 'response_format':
        return (code in ('unsupported_value', 'invalid_value') and
                ('not supported' in message or 'unsupported' in message)
                and ('json_object' in message or 'json_schema' in message))
    return False


def next_reasoning_mode(config, current):
    if current == 'reasoning_effort':
        host = urllib.parse.urlsplit(config['base_url']).hostname or ''
        # Avoid guessing a nonstandard extension on unknown gateways.
        if 'deepseek' in config['model'].lower() or host == 'api.deepseek.com':
            return 'thinking'
        return 'omit'
    if current == 'thinking':
        return 'omit'
    return None


def model_endpoints(settings):
    """Only explicitly configured, role-compatible Chat Completions endpoints."""
    required = {'provider','base_url','api_key_env','model','response_format','temperature',
                'timeout_seconds','max_retries','max_tokens'}
    if not isinstance(settings, dict) or not required <= set(settings) or set(settings)-required-{'reasoning','fallbacks'}:
        raise ValueError('Unexpected model fields; configure credentials with api_key_env')
    fallbacks = settings.get('fallbacks', [])
    if not isinstance(fallbacks, list):
        raise ValueError('fallbacks must be a list of explicitly configured endpoints')
    primary = {key:value for key,value in settings.items() if key!='fallbacks'}
    chain = [primary]
    for fallback in fallbacks:
        if (not isinstance(fallback, dict) or not {'base_url','model','api_key_env'} <= set(fallback)
                or set(fallback)-required-{'reasoning'}):
            raise ValueError('Each fallback requires base_url, model and api_key_env; nested fallbacks are unsupported')
        chain.append(dict(primary, **fallback))
    for config in chain:
        if config.get('reasoning') is None:
            config.pop('reasoning', None)
        elif (not isinstance(config['reasoning'], str) or
              config['reasoning'] not in REASONING_LEVELS):
            raise ValueError('reasoning must be one of: none, minimal, low, medium, high, xhigh, max')
        if any(not isinstance(config[name], str) or not config[name].strip()
               for name in ('base_url','model','api_key_env')):
            raise ValueError('Endpoint, model and credential variable must be nonempty strings')
        url = urllib.parse.urlsplit(config['base_url'])
        if (url.scheme not in ('http','https') or not url.hostname or url.username or url.password
                or url.query or url.fragment):
            raise ValueError('API base_url must be an HTTP(S) URL without credentials/query')
        if config['provider']!='openai_compatible':
            raise ValueError('Only openai_compatible is supported')
        if config['response_format'] not in ('json_schema','json_object'):
            raise ValueError('response_format must be json_schema or json_object')
        if (not isinstance(config['timeout_seconds'],(int,float)) or isinstance(config['timeout_seconds'],bool)
                or not math.isfinite(config['timeout_seconds']) or config['timeout_seconds']<=0
                or type(config['max_retries']) is not int or not 0<=config['max_retries']<=5
                or type(config['max_tokens']) is not int or config['max_tokens']<1
                or not isinstance(config['temperature'],(int,float)) or isinstance(config['temperature'],bool)
                or not math.isfinite(config['temperature']) or not 0<=config['temperature']<=2):
            raise ValueError('Invalid API model, timeout or retries')
    return chain


class APIClient:
    def __init__(self, endpoints, cache_dir):
        self.endpoints = endpoints
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.keys = {}
        # Per-worker capability inference survives episode resets.
        self.reasoning_modes = {}
        # Learned wire parameters are scoped to the endpoint + model, never
        # confused with navigation state, schema validity or HTTP auth failures.
        self.request_modes = {}
        for role in ROLES:
            for index,config in enumerate(model_endpoints(endpoints[role])):
                key = os.environ.get(config['api_key_env'])
                if not key:
                    raise ValueError('Set environment variable ' + config['api_key_env'])
                self.keys[(role,index)] = key
        # Explicit construction before the first episode is useful for preflight.
        self.begin_episode('preflight', 1, None)

    def begin_episode(self, episode_id, budget, log_path):
        if budget < 1:
            raise ValueError('API request budget must be positive')
        self.episode_id, self.budget, self.log_path = episode_id, budget, log_path
        self.calls, self.cache_hits, self.seconds = 0, 0, 0.0
        self.usage = {}
        self.requests_by_purpose, self.hits_by_purpose = {}, {}
        self.latencies, self.retries, self.normalizations = [], 0, 0
        self.by_role = {role: self._empty_stats() for role in ROLES}
        self.trace_context = {}
        self.completion_requests = 0
        self.by_endpoint = {}
        self.replay_recorder = None

    def availability(self):
        if self.calls >= self.budget:
            return 'episode_api_budget_exhausted'
        return 'available'

    def budget_state(self):
        return dict(total=self.budget, used=self.calls, remaining=max(0, self.budget - self.calls))

    def stats(self):
        return dict(requests=self.calls, cache_hits=self.cache_hits,
                    calls_saved=self.cache_hits,
                    cache_hit_rate=self.cache_hits / max(1, self.calls + self.cache_hits),
                    requests_by_purpose=self.requests_by_purpose,
                    cache_hits_by_purpose=self.hits_by_purpose,
                    seconds=self.seconds, usage=self.usage,
                    latency=latency_stats(self.latencies), retries=self.retries,
                    local_normalizations=self.normalizations, budget=self.budget_state(),
                    by_role=self.by_role, by_endpoint=self.by_endpoint)

    @staticmethod
    def _empty_stats():
        return dict(requests=0, seconds=0.0, retries=0, errors=0,
            validation_errors=0, timeouts=0, cache_hits=0, usage={})

    def _account(self, role, record):
        endpoint_key = role+'/'+record['endpoint_id']
        endpoint = self.by_endpoint.setdefault(endpoint_key, dict(self._empty_stats(),
            model=record['model'],base_url=record['base_url']))
        for group in (self.by_role[role], endpoint):
            for key,value in dict(requests=1,seconds=record['seconds'],
                    retries=int(record['attempt']>0 or record['endpoint_index']>0 or record['format_retry']>0),
                    errors=int(not record['success']),
                    validation_errors=int(record.get('error_type')=='ValidationError'),
                    timeouts=int(record.get('error')=='request_timeout')).items():
                group[key] += value
            for key,value in usage_counts(record.get('usage')).items():
                group['usage'][key] = group['usage'].get(key,0)+value

    def _log(self, record):
        if self.log_path is not None:
            messages=record.get('messages', [])
            record['input_text_chars']=sum(len(m['content']) if isinstance(m.get('content'),str) else
                sum(len(c.get('text','')) for c in m.get('content',[])) for m in messages)
            record['prior_conversation_text_chars']=(
                messages[1].get('prior_conversation_text_chars',0)
                if len(messages)>1 else 0)
            record['prior_conversation_image_count']=(
                messages[1].get('prior_conversation_image_count',0)
                if len(messages)>1 else 0)
            record['prior_conversation_turns']=(
                messages[1].get('prior_conversation_turns',0)
                if len(messages)>1 else 0)
            record['input_text_chars']+=record['prior_conversation_text_chars']
            record['system_chars']=sum(len(m['content']) for m in messages if m['role']=='system')
            record['request_text_plus_schema_chars'] = record['input_text_chars'] + record.get('native_schema_chars', 0)
            picture_hashes=messages[1].get('image_sha256',[]) if len(messages)>1 else []
            record['image_count']=(len(picture_hashes) if isinstance(picture_hashes,list)
                                   else int(bool(picture_hashes)))+record[
                                       'prior_conversation_image_count']
            line = canonical(dict(episode_id=self.episode_id, **self.trace_context, **record))
            for key in self.keys.values():
                line = line.replace(key, '[REDACTED]')
            with Path(self.log_path).open('a') as handle:
                handle.write(line + '\n')
            recorder = getattr(self, 'replay_recorder', None)
            if recorder is not None:
                try:
                    recorder.api_logged(dict(self.trace_context, **record))
                except OSError:
                    # Replay is a passive diagnostic, not an execution gate.
                    pass
            # api.jsonl stays the single API ledger: no duplicate API dumps.

    def _inflight(self, record, status):
        """Small, secret-free live marker for requests not yet in api.jsonl."""
        if self.log_path is None:
            return
        fields = ('role', 'purpose', 'request_number', 'attempt_number', 'model',
                  'endpoint_id', 'request_phase', 'effective_timeout_seconds',
                  'error')
        atomic_json(Path(self.log_path).with_name('api-inflight.json'),
            dict(status=status, updated_at_epoch=time.time(),
                 **{key:record[key] for key in fields if key in record}))

    def _request(self, request, timeout, record):
        with request_deadline(timeout):
            start = time.monotonic()
            record['request_phase'] = 'connect_and_headers'
            self._inflight(record, 'in_progress')
            try:
                response = urllib.request.urlopen(request, timeout=timeout)
            except urllib.error.HTTPError as error:
                # Error bodies can drip indefinitely too; use the same deadline.
                record['http_status'] = error.code
                if error.fp is not None:
                    record['request_phase'] = 'http_error_body'
                    try:
                        record['error_detail'] = error.read(8192).decode('utf-8', errors='replace')
                    except (socket.timeout, TimeoutError):
                        raise
                    except OSError:
                        record['error_detail'] = 'unreadable_error_response'
                    finally:
                        error.close()
                raise
            record['headers_seconds'] = time.monotonic() - start
            with response:
                record['request_phase'] = 'response_body'
                self._inflight(record, 'in_progress')
                start = time.monotonic()
                payload = response.read()
                record['body_seconds'] = time.monotonic() - start
            record['request_phase'] = 'decode_response'
            self._inflight(record, 'in_progress')
            result = json.loads(payload, parse_constant=reject_json_constant)
            record['request_phase'] = 'response_received'
            return result

    def complete(self, role, purpose, system, context, schema, image_url=None, cache=False, validate=None,
                 normalize=None, timeout_seconds=None, format_retry=0, result_schema=None, cache_context=None,
                 retry_safe=False, protocol=None, validation_schema=None,
                 conversation_messages=None):
        # Only pure inference callers opt in. No environment tool executes until
        # a single response has been returned and independently validated.
        self.completion_requests += 1
        request_id = self.completion_requests
        request_start = self.calls
        failures = []
        for endpoint_index,config in enumerate(model_endpoints(self.endpoints[role])):
            try:
                return self._complete_endpoint(role,purpose,system,context,schema,config,endpoint_index,
                    request_id,request_start,image_url=image_url,cache=cache,validate=validate,normalize=normalize,
                    timeout_seconds=timeout_seconds,format_retry=format_retry,result_schema=result_schema,
                    cache_context=cache_context,retry_safe=retry_safe,protocol=protocol,
                    previous_endpoint_failures=failures,validation_schema=validation_schema,
                    conversation_messages=conversation_messages)
            except BudgetExceeded:
                raise
            except APIError as error:
                failure = getattr(error,'request_context',dict(error=str(error),category=error.category))
                failures.append(dict(failure))
                error.request_context = dict(failure,endpoint_failures=failures[:],
                    physical_attempts=self.calls-request_start,completion_request_id=request_id)
                if not retry_safe or not getattr(error,'endpoint_unavailable',False):
                    raise
                final_error = error
        raise final_error

    def _complete_endpoint(self, role, purpose, system, context, schema, config, endpoint_index,
                 request_id, request_start, image_url=None, cache=False, validate=None, normalize=None,
                 timeout_seconds=None, format_retry=0, result_schema=None, cache_context=None,
                 retry_safe=False, protocol=None, previous_endpoint_failures=None, validation_schema=None,
                 conversation_messages=None):
        endpoint_id = 'primary' if endpoint_index==0 else 'fallback_%d' % endpoint_index
        validator = Draft7Validator(validation_schema or schema)
        result_validator = Draft7Validator(result_schema or validation_schema or schema)
        wire_schema = provider_schema(schema)
        if config['response_format'] != 'json_schema':
            system += ('\n'+protocol if protocol else '\nReturn only JSON satisfying this schema: '+canonical(wire_schema))
        messages = [{'role': 'system', 'content': system},
                    {'role': 'user', 'content': canonical(context)}]
        logged_messages = json.loads(canonical(messages))
        context_chars = len(canonical(context))
        if image_url is not None:
            images = image_url if isinstance(image_url, list) else [image_url]
            messages[1]['content'] = [
                {'type': 'text', 'text': canonical(context)}] + [
                {'type': 'image_url', 'image_url': {'url': value}} for value in images]
            hashes = [hashlib.sha256(value.encode()).hexdigest() for value in images]
            logged_messages[1]['image_sha256'] = hashes if isinstance(image_url, list) else hashes[0]
        # Re-send the task-local multimodal dialogue on each stateless request.
        # Never persist raw historical image base64 payloads in the audit logs.
        if conversation_messages:
            if (not isinstance(conversation_messages,list) or
                    any(not isinstance(turn,dict) or
                        turn.get('role') not in ('user','assistant','tool')
                        for turn in conversation_messages)):
                raise APIError('invalid_observation_conversation',category='structured_field')
            messages=[messages[0]]+copy.deepcopy(conversation_messages)+[messages[1]]
            # The audit deliberately omits historical base64 image payloads.
            # Still count the REAL earlier dialogue sent over the wire:
            # the old input_text_chars/image_count silently excluded it.
            history_text_chars=0
            history_image_count=0
            for turn in conversation_messages:
                content=turn.get('content')
                if isinstance(content,str):
                    history_text_chars+=len(content)
                elif isinstance(content,list):
                    for part in content:
                        if not isinstance(part,dict):
                            continue
                        if isinstance(part.get('text'),str):
                            history_text_chars+=len(part['text'])
                        if part.get('type')=='image_url':
                            history_image_count+=1
            logged_messages[1]['prior_conversation_turns']=len(
                conversation_messages)
            logged_messages[1]['prior_conversation_text_chars']=(
                history_text_chars)
            logged_messages[1]['prior_conversation_image_count']=(
                history_image_count)
        response_format = {'type': config['response_format']}
        if config['response_format'] == 'json_schema':
            response_format['json_schema'] = dict(name=purpose, strict=True, schema=wire_schema)
        exposed_names = [
            item.get('properties',{}).get('tool',{}).get('const')
            for item in schema.get('oneOf',[])
        ] if role == 'navigation_agent' and isinstance(schema,dict) else []
        exposed_names = [name for name in exposed_names if isinstance(name,str)]
        request_metrics = dict(system_chars=len(system),
            tool_protocol_chars=len(protocol or ''),
            context_chars_before_encoding=context_chars,
            offered_tool_count=len(exposed_names) if exposed_names else None,
            offered_tool_names=exposed_names if exposed_names else None,
            native_schema_chars=len(canonical(response_format))
            if config['response_format'] == 'json_schema' else 0,
            schema_channels=['response_format'] if config['response_format']=='json_schema' else
                ['compact_protocol'] if protocol else ['system'])
        reasoning_key = (config['base_url'], config['model'])
        learned_params = self.request_modes.get(reasoning_key, {})
        # The official OpenAI endpoint uses max_completion_tokens for reasoning
        # models; legacy OpenAI-compatible providers often still use max_tokens.
        # Unknown gateways are negotiated only upon explicit parameter rejection.
        host = urllib.parse.urlsplit(config['base_url']).hostname or ''
        token_field = learned_params.get('token_field',
                                        'max_completion_tokens' if host == 'api.openai.com'
                                        else 'max_tokens')
        body = dict(model=config['model'], messages=messages)
        if not learned_params.get('omit_response_format', False):
            body['response_format'] = response_format
        elif config['response_format'] == 'json_schema':
            # Once this endpoint rejects native schemas, subsequent requests
            # still need the full task-specific schema in their system text.
            addition = ('\nReturn only JSON satisfying this schema: ' +
                        canonical(wire_schema))
            messages[0]['content'] += addition
            logged_messages[0]['content'] += addition
            request_metrics['system_chars'] = len(messages[0]['content'])
            request_metrics['schema_channels'] = ['system_fallback']
            request_metrics['native_schema_chars'] = 0
        if not learned_params.get('omit_temperature', False):
            body['temperature'] = config['temperature']
        body[token_field] = config['max_tokens']
        requested_reasoning = config.get('reasoning')
        reasoning_mode = (self.reasoning_modes.get(reasoning_key, 'reasoning_effort')
                          if requested_reasoning is not None else 'omit')
        if requested_reasoning is not None:
            body.update(reasoning_payload(requested_reasoning, reasoning_mode))
        # Capture the assembled multi-turn wire input without Base64 in logs.
        # Recorder failure must never change the model's actual request.
        replay = None
        recorder = getattr(self, 'replay_recorder', None)
        if recorder is not None:
            try:
                replay = recorder.request_snapshot(messages, context, body)
            except (OSError, ValueError, TypeError, KeyError) as error:
                replay = dict(version=1, incomplete=True,
                              error='capture_' + type(error).__name__)
        def cache_identity():
            signature = dict(version=PROMPT_VERSION, role=role,
                             endpoint=config['base_url'], request=body)
            if cache and cache_context is not None:
                cached_body = json.loads(canonical(body))
                content = cached_body['messages'][1]['content']
                if isinstance(content, list):
                    content[0]['text'] = canonical(cache_context)
                else:
                    cached_body['messages'][1]['content'] = canonical(cache_context)
                signature['request'] = cached_body
            if result_schema is not None or validation_schema is not None:
                signature['result_schema'] = provider_schema(result_schema or validation_schema)
            digest = hashlib.sha256(canonical(signature).encode()).hexdigest()
            return digest, self.cache_dir / (digest + '.json')

        digest, cache_path = cache_identity()
        if cache and cache_path.exists():
            try:
                result = json.loads(cache_path.read_text(), parse_constant=reject_json_constant)
                result_validator.validate(result)
                if validate is not None:
                    validate(result)
            except (ValueError, ValidationError, APIError):
                raise APIError('invalid_' + purpose + '_cache') from None
            self.cache_hits += 1
            self.by_role[role]['cache_hits'] += 1
            endpoint_stats = self.by_endpoint.setdefault(role+'/'+endpoint_id,dict(self._empty_stats(),
                model=config['model'],base_url=config['base_url']))
            endpoint_stats['cache_hits'] += 1
            self.hits_by_purpose[purpose] = self.hits_by_purpose.get(purpose, 0) + 1
            self._log(dict(request_metrics, role=role, purpose=purpose, context_chars=context_chars,
                           cache_key=digest, cache_hit=True,
                           messages=logged_messages, result=result, model=config['model'],
                           endpoint_id=endpoint_id,endpoint_index=endpoint_index,base_url=config['base_url'],
                           completion_request_id=request_id,success=True, seconds=0.0,
                           replay=replay))
            return result
        retries = config['max_retries']
        timeout = config['timeout_seconds'] if timeout_seconds is None else min(config['timeout_seconds'], timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0:
            raise APIError('request_timeout')
        # Parameter negotiation is separate from retry_safe transport attempts.
        # Only a clearly rejected request can be renegotiated without executing
        # an action or repeating an accepted model completion.
        max_attempts = retries + 1 + (2 if requested_reasoning is not None else 0) + 3
        transient_retries = 0
        negotiated_fields = set()
        for attempt in range(max_attempts):
            available = self.availability()
            if available != 'available':
                raise BudgetExceeded(available)
            self.calls += 1
            self.retries += int(attempt > 0 or endpoint_index > 0 or format_retry > 0)
            self.requests_by_purpose[purpose] = self.requests_by_purpose.get(purpose, 0) + 1
            start = time.monotonic()
            deadline = start + timeout
            record = dict(request_metrics, role=role, purpose=purpose, context_chars=context_chars,
                          cache_key=digest, cache_hit=False,
                          request_number=self.calls,attempt=attempt,attempt_number=attempt+1,
                          completion_request_id=request_id,completion_attempt=self.calls-request_start,
                          endpoint_id=endpoint_id,endpoint_index=endpoint_index,base_url=config['base_url'],
                          max_attempts=max_attempts,retry_safe=retry_safe,
                          format_retry=format_retry,
                          model=config['model'], messages=logged_messages, replay=replay, request_type=purpose,
                          repair_attempt=attempt,
                          validation_error_type=None, validation_field=[], repair_result='not_needed',
                          fallback_reason=(previous_endpoint_failures[-1]['error']
                              if previous_endpoint_failures else None),final_decision=None,
                          request_token_field=token_field,
                          request_temperature_mode=('omitted_provider_default'
                              if 'temperature' not in body else 'configured'),
                          request_response_format_mode=('local_validation_only'
                              if 'response_format' not in body else config['response_format']),
                          reasoning=requested_reasoning, reasoning_wire_mode=reasoning_mode,
                          reasoning_fidelity=('exact' if reasoning_mode=='reasoning_effort' else
                              'boolean_only' if reasoning_mode=='thinking' else
                              'not_controllable' if requested_reasoning is not None else 'not_requested'),
                          timeout_seconds=config['timeout_seconds'],
                          effective_timeout_seconds=timeout,
                          deadline_scope='physical_attempt')
            retryable = False
            compatibility_retry = False
            next_mode = None
            compatibility_change = None
            endpoint_unavailable = False
            record['request_phase'] = 'preparing_request'
            self._inflight(record, 'in_progress')
            try:
                # Preserve insertion order of JSON Schema properties on the
                # wire. canonical() remains the stable, sorted signature and
                # audit formatter; sorting the outgoing body used to move the
                # model's decision fields ahead of its evidence fields.
                payload = json.dumps(body,ensure_ascii=False,
                                     allow_nan=False).encode()
                record['request_body_sha256'] = hashlib.sha256(payload).hexdigest()
                request = urllib.request.Request(
                    config['base_url'].rstrip('/') + '/chat/completions',
                    data=payload,
                    headers={'Content-Type': 'application/json',
                             'Authorization': 'Bearer ' + self.keys[(role,endpoint_index)]})
                remaining = deadline - time.monotonic()
                record['remaining_deadline_seconds'] = max(0, remaining)
                if remaining <= 0:
                    raise RequestDeadlineExceeded('total_request_deadline_exceeded')
                envelope = self._request(request, remaining, record)
                choice = envelope['choices'][0]
                message = choice['message']
                record['raw_response'] = message.get('content')
                record['response_model'] = envelope.get('model')
                record['usage'] = envelope.get('usage')
                record['finish_reason'] = choice.get('finish_reason')
                # Count reported usage even when a provider returned unusable JSON.
                for name, count in usage_counts(envelope.get('usage')).items():
                    self.usage[name] = self.usage.get(name, 0) + count
                if message.get('refusal') or choice.get('finish_reason') != 'stop':
                    # Repeating a refused/truncated request with the same limit won't repair it.
                    retryable = False
                    raise APIError('refusal_or_incomplete_response')
                result, normalization = decode_response(
                    message['content'],
                    allow_identical_duplicates=(
                        role=='visual_perception' and
                        purpose=='vlm_guided_observation'))
                if normalization:
                    record['response_normalization'] = normalization
                errors = list(validator.iter_errors(result))
                structural = errors
                if structural:
                    raise specific_schema_error(structural[0])
                if normalize is not None:
                    original = result
                    result = normalize(result)
                    if result != original:
                        self.normalizations += 1
                        record['repair_result'] = 'locally_normalized'
                # A navigation-specific validator reports the exact bad
                # tool parameter before the generic tagged-union schema
                # produces an unhelpful top-level oneOf error. It still
                # validates the full registered tool schema and never
                # repairs or executes the malformed action.
                if role=='navigation_agent' and validate is not None:
                    validate(result)
                result_validator.validate(result)
                if role!='navigation_agent' and validate is not None:
                    validate(result)
                if time.monotonic() >= deadline:
                    raise RequestDeadlineExceeded('total_request_deadline_exceeded')
                if record.get('response_normalization') and record['repair_result'] != 'locally_normalized':
                    self.normalizations += 1
                    record['repair_result'] = 'locally_normalized'
                record['result'] = result
                record['final_decision'] = result.get('decision') if isinstance(result, dict) else None
                if record['repair_result'] == 'not_needed':
                    record['repair_result'] = 'repaired' if attempt else 'not_needed'
            except urllib.error.HTTPError as error:
                retryable = error.code in (408, 429) or 500 <= error.code <= 599
                record['error'] = 'http_' + str(error.code)
                record['error_type'] = type(error).__name__
                record['http_status'] = error.code
                # _request already read and closed the body under its deadline.
                # Redact before passing diagnostics to Runtime, not only on disk.
                detail = record.get('error_detail', 'provider_error_body_unavailable')[:2048]
                for key in self.keys.values():
                    detail = detail.replace(key, '[REDACTED]')
                record['error_detail'] = detail
                protocol_unsupported = (error.code==400 and config['response_format']=='json_schema'
                    and any(word in detail.lower() for word in ('json_schema','response_format')))
                endpoint_unavailable = retryable or error.code in (401,403,404) or protocol_unsupported
                retryable = retryable and retry_safe
                record['validation_error_type'] = ('protocol_unsupported' if protocol_unsupported else
                    'transport' if endpoint_unavailable else 'http_request')
                record['repair_result'] = 'failed'
                field = ('reasoning_effort' if reasoning_mode == 'reasoning_effort'
                         else 'thinking' if reasoning_mode == 'thinking' else None)
                if (error.code == 400 and requested_reasoning is not None and field
                        and unsupported_reasoning_parameter(detail, field)):
                    next_mode = next_reasoning_mode(config, reasoning_mode)
                    if next_mode is not None:
                        # The server explicitly rejected the request parameter.
                        compatibility_retry = True
                        retryable = True
                        endpoint_unavailable = False
                        record['reasoning_adaptation'] = (
                            reasoning_mode + '_unsupported_try_' + next_mode)
                        record['validation_error_type'] = 'protocol_unsupported'
                        compatibility_change = ('reasoning', next_mode)
                # Other parameter families are negotiated independently:
                # no implicit unlimited output and no silent changes to
                # model, multimodal input, credentials or message structure.
                if error.code == 400 and not compatibility_retry:
                    if ('token_field' not in negotiated_fields
                            and unsupported_chat_parameter(detail, token_field)):
                        compatibility_change = ('token_field',
                            'max_tokens' if token_field == 'max_completion_tokens'
                            else 'max_completion_tokens')
                    elif ('temperature' in body and
                            'temperature' not in negotiated_fields and
                            unsupported_chat_parameter(detail, 'temperature')):
                        compatibility_change = ('omit_temperature', True)
                    elif ('response_format' in body and
                            'response_format' not in negotiated_fields and
                            unsupported_chat_parameter(detail, 'response_format')):
                        compatibility_change = ('omit_response_format', True)
                    if compatibility_change is not None:
                        compatibility_retry = True
                        retryable = True
                        endpoint_unavailable = False
                        record['request_parameter_adaptation'] = (
                            compatibility_change[0] + ':' +
                            str(compatibility_change[1]))
                        record['validation_error_type'] = 'protocol_unsupported'
            except RequestDeadlineUnavailable as error:
                retryable = False
                record.update(error=str(error), validation_error_type='transport', repair_result='failed')
            except (RequestDeadlineExceeded, urllib.error.URLError, socket.timeout, TimeoutError, OSError) as error:
                reason = error.reason if isinstance(error, urllib.error.URLError) else error
                timed_out = isinstance(reason, (socket.timeout, TimeoutError, RequestDeadlineExceeded))
                record['error'] = 'request_timeout' if timed_out else 'transport_error'
                record['error_type'] = type(error).__name__
                record['error_reason_type'] = type(reason).__name__
                record['error_detail'] = str(reason)[:2048]
                if timed_out:
                    record['timeout_scope'] = ('total_request' if isinstance(reason, RequestDeadlineExceeded)
                                               else 'socket_operation')
                record['validation_error_type'] = 'transport_timeout' if timed_out else 'transport'
                record['repair_result'] = 'failed'
                record['upstream_completion_unknown'] = timed_out
                known_permanent_http = record.get('http_status') in (400,405,413,422)
                if known_permanent_http:
                    record['validation_error_type'] = 'http_request'
                endpoint_unavailable = not known_permanent_http
                retryable = retry_safe and endpoint_unavailable
            except (ValueError, KeyError, IndexError, TypeError, ValidationError, APIError) as error:
                record['error'] = 'invalid_model_response'
                record['error_type'] = type(error).__name__
                if isinstance(error, ValidationError):
                    record['validation_path'] = list(error.absolute_path)
                    record['validation_field'] = list(error.absolute_path)
                    record['validation_error_type'] = 'structured_field'
                    record['validation_rule'] = error.validator
                    actual = len(error.instance) if isinstance(error.instance, (str, list, dict)) else None
                    record['actual_length'] = actual
                    record['validation_limit'] = error.validator_value if isinstance(error.validator_value, (str, int, float, bool)) else None
                    record['error_detail'] = '%s: %s requires %s; actual length %s' % (
                        '.'.join(map(str, error.absolute_path)) or 'response', error.validator,
                        str(error.validator_value)[:200], actual)
                    if error.validator == 'required' and isinstance(error.instance, dict):
                        record['missing_fields'] = [k for k in error.validator_value if k not in error.instance]
                        record['validation_field'] += record['missing_fields'][:1]
                        record['error_detail'] = 'Missing required fields at %s: %s' % (
                            '.'.join(map(str, error.absolute_path)) or 'response', ', '.join(record['missing_fields']))
                else:
                    record['error_detail'] = str(error)[:1000]
                    record['validation_error_type'] = getattr(error, 'validation_error_type',
                        'json_syntax' if isinstance(error, json.JSONDecodeError) else 'structured_field')
                    record['validation_field'] = getattr(error, 'validation_field', [])
                record['repair_result'] = 'failed'
                retryable = False
                record['retry_skipped'] = 'format_error_requires_fresh_role_request'
            finally:
                elapsed = time.monotonic() - start
                record['seconds'] = elapsed
                self.seconds += elapsed
                self.latencies.append(elapsed)
                record['success'] = 'error' not in record
                self._account(role, record)
                record['retry_scheduled'] = bool(
                    'error' in record and retryable and attempt < max_attempts-1
                    and (compatibility_retry or transient_retries < retries)
                    and self.availability() == 'available')
                self._log(record)
                self._inflight(record, 'completed')
            if 'error' not in record:
                if cache:
                    atomic_json(cache_path, result)
                return result
            if not record['retry_scheduled']:
                if compatibility_retry and self.availability() != 'available':
                    raise BudgetExceeded(self.availability())
                failure = APIError(record['error'], category=record.get('validation_error_type'))
                failure.endpoint_unavailable = endpoint_unavailable
                failure.validation_error = {k:record[k] for k in ('validation_field', 'validation_rule',
                    'validation_limit', 'actual_length', 'missing_fields') if k in record}
                failure.request_context = {k:record[k] for k in ('role', 'purpose', 'request_number',
                    'request_phase', 'error', 'error_type', 'error_detail', 'timeout_scope',
                    'effective_timeout_seconds','validation_error_type','validation_field','http_status',
                    'endpoint_id','endpoint_index','base_url','model','attempt_number','completion_attempt',
                    'completion_request_id','upstream_completion_unknown') if k in record}
                raise failure
            if compatibility_retry:
                kind, value = compatibility_change
                if kind == 'reasoning':
                    reasoning_mode = value
                    self.reasoning_modes[reasoning_key] = reasoning_mode
                    body.pop('reasoning_effort', None)
                    body.pop('thinking', None)
                    body.update(reasoning_payload(requested_reasoning, reasoning_mode))
                else:
                    negotiated_fields.add('token_field' if kind == 'token_field' else
                                          'temperature' if kind == 'omit_temperature'
                                          else 'response_format')
                    self.request_modes.setdefault(reasoning_key, {})[kind] = value
                    if kind == 'token_field':
                        body[value] = body.pop(token_field)
                        token_field = value
                    elif kind == 'omit_temperature':
                        body.pop('temperature', None)
                    elif kind == 'omit_response_format':
                        body.pop('response_format', None)
                        # json_object already embeds the full schema in the
                        # system message. json_schema needs an equivalent text
                        # contract once provider-side native enforcement is lost.
                        if config['response_format'] == 'json_schema':
                            addition = ('\nReturn only JSON satisfying this schema: ' +
                                        canonical(wire_schema))
                            messages[0]['content'] += addition
                            logged_messages[0]['content'] += addition
                            request_metrics['system_chars'] = len(messages[0]['content'])
                            request_metrics['schema_channels'] = ['system_fallback']
                            request_metrics['native_schema_chars'] = 0
                digest, cache_path = cache_identity()
                if recorder is not None:
                    try:
                        replay = recorder.request_snapshot(messages, context, body)
                    except (OSError, ValueError, TypeError, KeyError) as error:
                        replay = dict(version=1, incomplete=True,
                                      error='capture_' + type(error).__name__)
            else:
                transient_retries += 1
                # Backoff is outside the next attempt's fresh request deadline.
                time.sleep(min(2 ** (transient_retries-1), 4))
        raise APIError('request_failed')


def latency_stats(values):
    if not values:
        return dict(count=0, mean=None, p50=None, p95=None, maximum=None)
    ordered = sorted(values)
    def percentile(fraction):
        index = (len(ordered) - 1) * fraction
        low = int(index)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)
    return dict(count=len(values), mean=sum(values) / len(values),
                p50=percentile(0.5), p95=percentile(0.95), maximum=ordered[-1])
