"""Model-only bearing projection. Canonical Memory and execution keep absolute yaw.

All serialized model-facing camera/goal headings are expressed as *signed
offsets from the robot's current camera yaw*: positive LEFT, negative RIGHT.
Historical orientation offsets are NOT executable movement bearings across
different executed-forward stances. Pitch, relative turns and sector widths
are never reinterpreted as absolute headings.
"""
import copy
import json
import math
import re

from .heading import relative_bearing

# Only source-bearing fields are transformed. Plain 'bearing_deg' and
# 'turn_degrees' are already local/relative and must not be transformed.
_ABSOLUTE_DEG_KEYS={
    'heading_deg','source_heading_deg','current_heading_deg',
    'last_heading_deg','heading_at_capture_deg','absolute_heading_deg',
    'goal_heading_deg','target_heading_deg','yaw_deg','source_yaw_deg',
    'current_yaw_deg','absolute_yaw_deg','camera_yaw_deg',
    'azimuth_deg','absolute_azimuth_deg',
}
_ABSOLUTE_RAD_KEYS={'heading','yaw'}  # Only for pose-like structures.
_ABSOLUTE_BIN_KEYS={'observed_heading_bins_deg','heading_bins_deg',
                    'observed_yaw_bins_deg'}

def _is_number(value):
    return (isinstance(value,(int,float)) and not isinstance(value,bool)
            and math.isfinite(value))

def _offset(angle, reference):
    return round(relative_bearing(float(angle)%360,reference),2)

def _view_label(key):
    if key in ('heading_deg','yaw_deg','azimuth_deg','camera_yaw_deg'):
        return 'view_offset_deg_from_current'
    if key in ('current_heading_deg','current_yaw_deg'):
        return 'current_view_offset_deg_from_current'
    if key in ('last_heading_deg',):
        return 'last_view_offset_deg_from_current'
    if key in ('source_heading_deg','source_yaw_deg'):
        return 'source_view_offset_deg_from_current'
    if key=='heading_at_capture_deg':
        return 'capture_view_offset_deg_from_current'
    if key.startswith('absolute_') or key.startswith('target_') or key.startswith('goal_'):
        return 'historical_target_offset_deg_from_current'
    return key+'_offset_from_current'

# Only explicitly heading-labelled numbers in historical/model narrative
# can be treated as world bearings. Do not guess that arbitrary "30 degrees"
# describes an absolute yaw; it might be a true relative turn or pitch.
_NARRATIVE_YAW=re.compile(
    r'(?<![\w])(?P<label>heading|yaw|azimuth|航向角?|朝向)'
    r'\s*(?:(?:is|at|of)\s+|[:=]\s*)?'
    r'(?P<deg>\d{1,3}(?:\.\d+)?)'
    r'\s*(?:°|degrees?\b|deg\b)?',re.I)
_KEEP_ORIGINAL_TEXT_KEYS={'instruction','full_instruction',
                          'mission_instruction','original_instruction'}

def _relative_narrative(text,reference):
    def replace(match):
        start=match.start()
        preceding=text[max(0,start-14):start].lower()
        if preceding.endswith('relative ') or preceding.endswith('signed '):
            return match.group(0)
        absolute=float(match.group('deg'))
        if absolute>360:
            return match.group(0)
        delta=_offset(absolute,reference)
        side=('LEFT' if delta>0 else 'RIGHT' if delta<0 else 'AHEAD')
        return ('view offset '+side+' '+format(abs(delta),'g')+
                ' degrees from current camera')
    return _NARRATIVE_YAW.sub(replace,text)


def project_model_angles(value, current_heading_deg):
    """Return a deep, read-only transformation for model wire contexts.

    Raw records remain byte-identical. All converted offsets are **viewpoint
    orientations** unless an already-source-validated same-stance candidate
    explicitly marks a local target. This helper never computes motor action
    arguments and never calls a model.
    """
    if not _is_number(current_heading_deg):
        raise ValueError('model_angle_reference_requires_finite_yaw_degrees')
    reference=float(current_heading_deg)%360

    def project(node, parent=''):
        if isinstance(node,list):
            return [project(v,parent) for v in node]
        if isinstance(node,str):
            return _relative_narrative(node,reference) if parent not in (
                _KEEP_ORIGINAL_TEXT_KEYS) else node
        if not isinstance(node,dict):
            return copy.deepcopy(node)
        result={}
        # Rebase any precomputed target bearing in a previous VLM session.
        # The raw session retains the true absolute target anchor internally.
        # A null bearing means another stance: NEVER make it executable.
        absolute=node.get('absolute_heading_deg')
        original_relative=node.get('relative_bearing_deg_from_current')
        if _is_number(absolute) and _is_number(original_relative):
            from .heading import direction_from_bearing
            delta=_offset(absolute,reference)
            result['relative_bearing_deg_from_current']=delta
            if 'relative_side_from_current' in node:
                side,angle=direction_from_bearing(delta)
                result['relative_side_from_current']=side
                result['relative_turn_deg_from_current']=angle
            if 'recommended_turn_direction' in node:
                side,angle=direction_from_bearing(delta)
                result['recommended_turn_direction']=side
                result['recommended_turn_degrees']=angle
        # Route continuity was computed against the then-current camera
        # heading. For rebased VLM conversation turns, refresh its delta.
        old_forward=node.get('selected_historical_forward')
        if isinstance(old_forward,dict) and _is_number(
                old_forward.get('heading_deg')):
            separation=abs(_offset(old_forward['heading_deg'],reference))
            if 'separation_from_historical_forward_deg' in node:
                result['separation_from_historical_forward_deg']=separation
                result['opposing_camera_heading']=separation>=150
                result['previous_forward_and_current_view_may_be_a_retrace_not_proven']=(
                    separation>=150)
        for key,v in node.items():
            if key in result:
                continue
            # A route waypoint may carry both the original absolute yaw
            # and an already recomputed same-stance relative target. Never
            # convert that historical yaw into a second movement suggestion.
            if key=='absolute_heading_deg' and (
                    'relative_bearing_deg_from_current' in node or
                    'relative_turn_deg_from_current' in node):
                continue
            if key in _ABSOLUTE_DEG_KEYS:
                if _is_number(v):
                    label=_view_label(key)
                    result[label]=_offset(v,reference)
                # Drop textual bearing conventions describing 0..360
                # absolute compass angles; they are NOT actionable input.
                continue
            if key in _ABSOLUTE_BIN_KEYS:
                if isinstance(v,(list,tuple)):
                    result['observed_view_offsets_deg_from_current']=sorted(
                        set(_offset(x,reference) for x in v if _is_number(x)))
                continue
            if key in _ABSOLUTE_RAD_KEYS and parent in (
                    'pose','source_pose','current_pose','start_pose','end_pose'):
                if _is_number(v):
                    result['view_offset_deg_from_current']=_offset(
                        math.degrees(v),reference)
                continue
            result[key]=project(v,key)
        return result

    projected=project(value)
    if isinstance(projected,dict):
        # The reference is deliberately a RELATIVE convention, never the
        # hidden world yaw used by the geometry engine.
        projected.setdefault('view_angle_convention',
            'Robot CURRENT RGB forward=0 degrees; positive=LEFT, '
            'negative=RIGHT; historical view offsets are NOT '
            'navigation bearings after a forward step.')
    return projected


def project_observation_conversation(messages, current_heading_deg):
    """Rebase raw VLM dialogue JSON to CURRENT view, without modifying Memory.

    Prior user requests include source absolute yaw metadata; previous VLM
    replies are also sanitized for clearly named historical yaw references.
    TURN/LOOK response fields are still the model's own relative decisions.
    """
    if not messages:
        return messages
    rendered=copy.deepcopy(messages)
    for message in rendered:
        content=message.get('content')
        if isinstance(content,str):
            try:
                decoded=json.loads(content)
            except (ValueError,TypeError):
                # Ordinary assistant prose may still refer to old yaw.
                message['content']=_relative_narrative(
                    content,float(current_heading_deg)%360)
            else:
                if isinstance(decoded,(dict,list)):
                    projected=project_model_angles(decoded,current_heading_deg)
                    if message.get('role')=='assistant' and isinstance(
                            projected,dict):
                        # A past VLM answer is not a new user directive.
                        # Keep its exact JSON if only the synthetic metadata
                        # differs, but rewrite any literal world bearings.
                        projected.pop('view_angle_convention',None)
                    if projected!=decoded:
                        message['content']=json.dumps(
                            projected,ensure_ascii=False,separators=(',',':'))
            continue
        if not isinstance(content,list):
            continue
        for part in content:
            if not isinstance(part,dict) or part.get('type')!='text':
                continue
            raw=part.get('text','')
            if not isinstance(raw,str):
                continue
            try:
                decoded=json.loads(raw)
            except (ValueError,TypeError):
                part['text']=_relative_narrative(
                    raw,float(current_heading_deg)%360)
            else:
                if isinstance(decoded,(dict,list)):
                    part['text']=json.dumps(
                        project_model_angles(decoded,current_heading_deg),
                        ensure_ascii=False,separators=(',',':'))
    return rendered
