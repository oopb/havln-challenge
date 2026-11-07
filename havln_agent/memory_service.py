"""Trusted RGB-D, place graph, task and dynamic memory with bounded retrieval."""
import copy
import math
import time

from .contexts import fields
from .memory_records import MemoryRecords, task_tokens
from .motion import registered_action
from .mission_history import mission_evidence




class MemoryService(MemoryRecords):
    def __init__(self, runtime):
        self.runtime=runtime
        super().__init__(runtime.camera)

    def reset(self, episode_id, instruction):
        super().reset(episode_id,instruction)
        self.places,self.keyframes,self.landmarks,self.routes={},{},{},{}
        self.dynamic,self.notes=[],[]
        self.visual_targets=[]
        self.current_place_id=None
        self.place_visit_step=0

    def note_observation(self, record):
        pose=record['pose']
        previous=self.current_place_id
        place=self.places.get(previous)
        def separation(place):
            return math.sqrt((pose['x']-place['position'][0])**2+(pose['z']-place['position'][1])**2
                +(pose.get('y',0)-place['height_y_m_est'])**2)
        # A failed RGB-D translation starts a new local reference epoch.
        # Do not silently retain a place node from the previous odometry chain
        # just because stale x/z estimates happen to be numerically nearby.
        if (place is None or place.get('motion_epoch') != pose.get('motion_epoch',0)
                or place.get('viewpoint_generation',0) != pose.get('viewpoint_generation',0)
                or separation(place) >= .5):
            near=[p for p in self.places.values() if p['motion_epoch']==pose.get('motion_epoch',0)
                and p.get('viewpoint_generation',0)==pose.get('viewpoint_generation',0)
                and separation(p) < .5]
            self.current_place_id=near[0]['id'] if near else 'P%d' % len(self.places)
            if self.current_place_id not in self.places:
                self.places[self.current_place_id]=dict(id=self.current_place_id,
                    position=[pose['x'],pose['z']],height_y_m_est=pose.get('y',0), confidence='rgbd_estimate_not_global_localization',
                    position_uncertainty_m=(pose['position_uncertainty']
                        if pose.get('position_coordinates_valid',True) else None),
                    motion_epoch=pose.get('motion_epoch',0),
                    viewpoint_generation=pose.get('viewpoint_generation',0),
                    observation_ids=[], headings_deg=[])
            place=self.places[self.current_place_id]
            if (previous is not None and previous != self.current_place_id
                    and self.places[previous]['motion_epoch'] == place['motion_epoch']):
                moves=[a for a in self.actions[self.place_visit_step:] if a['action']=='MOVE_FORWARD']
                key=previous+'->'+self.current_place_id
                self.routes[key]=dict(id=key,from_place=previous,to_place=self.current_place_id,
                    state='CANDIDATE',traversability='UNKNOWN',evidence_state='sensor_recorded',
                    motion_steps=[a['step'] for a in moves], source_observation_ids=[record['observation_id']],
                    confidence='rgbd_estimate_not_ground_truth')
                self.place_visit_step=len(self.actions)
        identifier=record['observation_id']
        record.update(place_id=self.current_place_id,pitch_deg=self.runtime.pitch_deg,evidence_state='sensor_recorded')
        if identifier not in place['observation_ids']:
            place['observation_ids'].append(identifier)
        heading=round(math.degrees(pose['heading']),4)
        if heading not in place['headings_deg']:
            place['headings_deg'].append(heading)
        self.keyframes[identifier]=dict(fields(record,('observation_id','input_file','rgb_hash','depth_hash',
            'captured_at','representative_keyframe','evidence_state','place_id','pitch_deg')),
            heading_deg=heading,findings=[])
        for route in self.routes.values():
            if route['state']=='EXECUTABLE' and route['source_observation_ids'] != [identifier]:
                route.update(state='CANDIDATE',traversability='UNKNOWN')

    def note_semantics(self, source, findings):
        identifier=source['observation_id']
        observation=self.keyframes[identifier]
        for text in findings:
            item=dict(finding=text,evidence_state='model_hypothesis',source_observation_ids=[identifier])
            if item not in observation['findings']:
                observation['findings'].append(item)
            if not self.current.get('revisited_static_view') or self.current['observation_id'] != identifier:
                observation['representative_keyframe']=True
            key=(observation['place_id'],text.lower())
            landmark=self.landmarks.setdefault(key,dict(name=text,place_id=observation['place_id'],
                evidence_state='model_hypothesis',source_observation_ids=[]))
            if identifier not in landmark['source_observation_ids']:
                landmark['source_observation_ids'].append(identifier)
        self.event('semantic_candidates_recorded',source=source,findings=findings)

    def note_active_observation_target(self, candidate, objective, target_types,
                                       goal=None):
        """Persist VLM-search evidence in the same typed visual target pool.

        This records an authentic RGB observation and an uncertain visual
        hypothesis. It never claims collision clearance, metric arrival or
        completed semantic navigation. The calling controller provides the
        actual source observation_id, yaw and camera pitch.
        """
        if not isinstance(candidate,dict):
            return None
        identifier=candidate.get('observation_id')
        if identifier not in self.keyframes:
            raise ValueError('active_visual_target_requires_real_observation')
        kind=candidate.get('target_kind','UNKNOWN')
        if kind not in ('ENTRY','APPROACH','TARGET','CUE'):
            return None
        selected=self.keyframes[identifier]
        status='CUE_ONLY' if kind=='CUE' else 'VISIBLE'
        current_goal=(copy.deepcopy(goal) if goal is not None
            else dict(objective=objective,target_types=list(target_types)))
        for existing in reversed(self.visual_targets):
            if (existing.get('selected_observation_id')==identifier
                    and existing.get('target_type')==candidate.get('target_type')
                    and existing.get('target_kind')==kind):
                return existing
        pose=self.runtime.motion_controller.pose()
        absolute=round(float(candidate.get('heading_deg',
            selected['heading_deg'])),2)
        relative=round((absolute-math.degrees(pose['heading'])+180)%360-180,2)
        record=dict(
            id='visual_target_%04d'%len(self.visual_targets),
            visual_source='vlm_active_observation',
            place_id=selected['place_id'],
            motion_epoch=pose.get('motion_epoch',0),
            viewpoint_generation=pose.get('viewpoint_generation',0),
            goal=current_goal,
            observation_objective=objective,
            target_type=candidate.get('target_type'),
            target_kind=kind,confidence=candidate.get('confidence'),
            expected_progress=candidate.get('expected_progress','UNKNOWN'),
            status=status,selected_observation_id=identifier,
            source_observation_ids=[identifier],
            absolute_heading_deg=absolute,
            pitch_deg=candidate.get('pitch_deg',selected.get('pitch_deg')),
            relative_heading_deg_at_review=relative,
            target_region=copy.deepcopy(candidate.get('box')),
            visual_evidence=(candidate.get('evidence') or '')[:400],
            uncertainty=(candidate.get('uncertainty') or '')[:300],
            missing_evidence=('confirm_actual_walkable_entry'
                if kind=='CUE' else ''),
            approach_candidates=[],
            evidence_state='historical_visual_hypothesis_not_clearance')
        self.visual_targets.append(record)
        self.event('active_visual_target_recorded',
                   target_id=record['id'],observation_id=identifier,
                   target_kind=kind,status=status)
        return record

    def note_panorama_target(self, visual, views, goal=None):
        """Retain provenance-linked goal/entry hypotheses across OBSERVE turns.

        Stored headings are historical estimates; they never imply a live
        collision clearance or actual reached landmark.
        """
        by_id={v['observation_id']:v for v in views}
        identifier=visual['selected_observation_id']
        if identifier not in by_id or identifier not in self.keyframes:
            raise ValueError('panorama_target_requires_real_observation')
        candidates=[]
        for item in visual.get('approach_candidates',[]):
            view=by_id.get(item['observation_id'])
            if view is None:
                raise ValueError('panorama_candidate_requires_real_observation')
            candidates.append(dict(item,
                absolute_heading_deg=view['absolute_heading_deg'],
                relative_heading_deg_at_review=view['relative_turn_deg_from_current']))
        main=by_id[identifier]
        record=dict(id='visual_target_%04d'%len(self.visual_targets),
            place_id=self.keyframes[identifier]['place_id'],
            motion_epoch=self.runtime.motion_controller.pose()['motion_epoch'],
            goal=copy.deepcopy(goal),
            status=visual.get('target_status','AMBIGUOUS'),
            selected_observation_id=identifier,
            source_observation_ids=[identifier]+[x['observation_id'] for x in candidates
                if x['observation_id']!=identifier],
            absolute_heading_deg=main['absolute_heading_deg'],
            relative_heading_deg_at_review=main['relative_turn_deg_from_current'],
            target_region=visual.get('target_region'),
            visual_evidence=visual['visible_evidence'],
            uncertainty=visual['uncertainty'],
            missing_evidence=visual.get('missing_evidence',''),
            approach_candidates=candidates,
            evidence_state='historical_visual_hypothesis_not_clearance')
        self.visual_targets.append(record)
        # Make target evidence searchable by observation ID, unlike ephemeral
        # panorama answers that previously left landmarks/groundings empty.
        label=goal if isinstance(goal,dict) else {}
        if record['status'] == 'VISIBLE':
            self.note_semantics(dict(observation_id=identifier),[
                str(label.get('target') or label.get('landmark') or
                    'navigation_target') + ': ' + visual['visible_evidence'][:300]])
        self.event('panorama_target_recorded',target_id=record['id'],
                   source_observation_ids=record['source_observation_ids'],
                   target_status=record['status'])
        return record

    def note_scene(self, scene, source):
        self.note_semantics(source,[v['name'] for v in scene.get('landmarks',[])])
        people=[dict(p,kind='person') for p in scene.get('humans',[])]+[
            dict(p,kind='uncertain_humanoid') for p in scene.get('uncertain_humanoids',[])]
        temporary=[dict(p,kind='temporary_obstacle') for p in scene.get('temporary_obstacles',[])]
        for item in people+temporary:
            self.dynamic.append(dict(id='dynamic_%04d' % len(self.dynamic),kind=item['kind'],
                place_id=self.keyframes[source['observation_id']]['place_id'],
                finding=item.get('activity',item.get('description','unknown activity')),
                box=item.get('box'),position=None,
                observed_at=source.get('revalidated_at',source['captured_at']),confidence='model_hypothesis',
                source_observation_ids=[source['observation_id']],usable_for_live_safety=False))

    def dynamic_summaries(self, place_id=None):
        now=time.monotonic()
        ttl=self.runtime.config['max_human_age_seconds']
        return [dict(fields(h,('id','kind','place_id','finding','position','confidence','source_observation_ids')),
            state='UNKNOWN' if now-h['observed_at']>ttl else 'RECENT_OBSERVATION',
            age_seconds=round(max(0,now-h['observed_at']),1),usable_for_live_safety=False)
            for h in self.dynamic if place_id is None or h['place_id']==place_id]

    def write_hypothesis(self, finding, source_observation_ids):
        if not source_observation_ids or any(i not in self.keyframes for i in source_observation_ids):
            raise ValueError('hypothesis_requires_existing_observation_sources')
        self.notes.append(dict(finding=finding,source_observation_ids=list(source_observation_ids),
            evidence_state='model_hypothesis'))
        self.event('hypothesis_written',**self.notes[-1])

    def note_candidates(self, paths):
        for path in paths:
            direct = path.get('direction_source') == 'navigation_agent'
            self.routes[path['id']]=dict(id=path['id'],waypoint_type=path['type'],from_place=self.current_place_id,to_place=None,
                state='CANDIDATE',
                traversability='UNKNOWN' if direct else path['first_step'],
                evidence_state='model_motion_proposal_unverified' if direct else 'geometry_supported',
                source_observation_ids=[path['source_observation_id']],bearing_deg=path['bearing_deg'],
                requires_live_revalidation=True)

    def note_validation(self, path, value):
        route=self.routes.get(path['id'])
        if route and value['status']=='OK' and not value['executable_segment']['turn_only']:
            evidence=(value.get('validation') or {}).get('forward_evidence_state')
            route.update(state='EXECUTABLE' if evidence=='EXECUTABLE' else 'CANDIDATE',
                evidence_state='geometry_supported' if evidence=='EXECUTABLE'
                    else 'depth_unconfirmed_motion_hypothesis',
                traversability='FREE' if evidence=='EXECUTABLE' else 'UNKNOWN',
                source_observation_ids=[self.current['observation_id']],
                validated_at=self.runtime.observation.captured_at)

    def note_motion(self):
        observations={o['observation_id']:o for o in self.observations}
        for route in self.routes.values():
            steps=route.get('motion_steps',[])
            if route['state']=='CANDIDATE' and route.get('to_place') and steps and all(
                    registered_action(self.actions[i],observations) for i in steps):
                route.update(state='TRAVERSED',evidence_state='motion_verified')

    def navigation_summary(self):
        """All-episode evidence, not measurements filtered to a subgoal."""
        motions=[]
        for action in self.actions:
            if action.get('action')!='MOVE_FORWARD':
                continue
            feedback=action.get('result') or {}
            m=feedback.get('rgbd_motion') or {}
            registered=m.get('status')=='ESTIMATED'
            d=m.get('translation_m') or {}
            motions.append(dict(step=action.get('step'),
                observation_id=feedback.get('observation_after'),
                heading_deg=round(math.degrees(
                    action.get('pose',{}).get('heading',0))%360,1),
                vertical_motion=m.get('vertical_motion','UNKNOWN') if registered else 'UNKNOWN',
                vertical_change_m_est=round(d.get('y_up',0),3) if registered else None,
                uncertainty_m=round(m.get('uncertainty_m',0),3) if registered else None,
                observed_frame_change=m.get('observed_frame_change')))
        failures=[]
        for item in reversed(self.tool_ledger):
            call=item.get('call') or {}
            result=item.get('result') or {}
            if call.get('tool') not in ('navigate_relative','navigate_local'):
                continue
            state=result.get('interrupted_status') or result.get('status')
            if state in ('BLOCKED','NEEDS_VIEW','REQUIRES_ATTENTION',
                         'INTERRUPTED','STALE_TARGET'):
                failures.append(dict(
                    tool_call_id=item.get('call_id'),status=state,
                    reason=result.get('reason'),
                    requested_motion=result.get('requested_motion') or
                        fields(call.get('args') or {},('turn_deg','distance_m')),
                    observation_id=result.get('end_observation_id')))
            if len(failures)==4:
                break
        return dict(total_forward_actions=len(motions),
            recent_forward_motion=motions[-8:],
            recent_failed_motion=failures,
            visual_target_memory=copy.deepcopy(self.visual_targets[-4:]),
            latest_actual_blocks=copy.deepcopy(self.evidence.obstacle_blocks[-4:]),
            note='Measured translation is uncertain RGB-D evidence, not ground truth.')

    def _routes(self):
        now=time.monotonic()
        values=copy.deepcopy(list(self.routes.values()))
        for item in values:
            if item['state']=='EXECUTABLE' and (now-item['validated_at']>self.runtime.config['max_depth_age_seconds']
                    or item['source_observation_ids'] != [self.current['observation_id']]):
                item.update(state='CANDIDATE',traversability='UNKNOWN')
        return values

    def traversed_routes(self):
        return [fields(r,('id','from_place','to_place','state','evidence_state','source_observation_ids'))
                for r in self._routes() if r['state']=='TRAVERSED']

    def recall_observation(self, place_id, heading_deg=None, landmark=None, top_k=3):
        self.queries+=1
        query=task_tokens(landmark or '')
        adjacent={r.get('to_place') for r in self._routes() if r['state']=='TRAVERSED' and r['from_place']==place_id}
        def score(item):
            words=task_tokens(' '.join(f['finding'] for f in item['findings']))
            tier=0 if item['place_id']==place_id else 1 if item['place_id'] in adjacent else 2
            heading=0 if heading_deg is None else abs((item['heading_deg']-heading_deg+180)%360-180)
            return tier,-len(query & words),heading,-int(bool(item['findings'])),-item['captured_at']
        options=[o for o in self.keyframes.values() if o['representative_keyframe'] or o['findings'] or heading_deg is not None]
        ranked=sorted(options,key=score)[:top_k]
        self.event('memory_retrieved',retrieval='observation',ids=[o['observation_id'] for o in ranked])
        return dict(observations=[dict(fields(o,('observation_id','place_id','heading_deg','pitch_deg','evidence_state')),
            findings=copy.deepcopy(o['findings'][:2])) for o in ranked],historical_depth_authorizes_motion=False)

    def query_spatial_memory(self, place_id, target=None, top_k=4):
        self.queries+=1
        query=task_tokens(target or '')
        items=[dict(kind='landmark',**fields(v,('name','place_id','evidence_state')),
            source_observation_ids=v['source_observation_ids'][-2:]) for v in self.landmarks.values()]
        items += [dict(kind='route',**fields(v,('id','from_place','to_place','state','traversability',
            'evidence_state','bearing_deg','source_observation_ids'))) for v in self._routes()]
        items += [dict(kind='visual_target',id=v['id'],place_id=v['place_id'],
            status=v['status'],goal=v['goal'],absolute_heading_deg=v['absolute_heading_deg'],
            approach_candidates=copy.deepcopy(v['approach_candidates']),
            source_observation_ids=v['source_observation_ids'],
            evidence_state=v['evidence_state']) for v in getattr(self,'visual_targets',[])]
        items += list(reversed(self.dynamic_summaries(place_id)))
        def score(item):
            local=item.get('place_id',item.get('from_place'))==place_id
            relevant=len(query & task_tokens(str(item.get('name',item.get('finding','')))))
            trusted=item.get('evidence_state') in ('motion_verified','task_verified')
            return not local,-relevant,-int(trusted),-int(item.get('state')=='RECENT_OBSERVATION')
        items=sorted(items,key=score)[:top_k]
        place=self.places.get(place_id)
        return dict(place=fields(place,('id','position','height_y_m_est','confidence','position_uncertainty_m','headings_deg')) if place else None,
            items=copy.deepcopy(items),dynamic_locations_authorize_motion=False)

    def query_task_memory(self):
        """On-demand expansion of the SAME canonical history given to both agents."""
        self.queries+=1
        return dict(
            mission_evidence=mission_evidence(
                self,self.runtime.motion_controller.pose(),
                max_observation_tasks=6),
            progress_revision_audit=[
                dict(step=v.get('step'),
                     progress_narrative=v.get('progress_narrative'),
                     navigation_intent=v.get('navigation_intent'),
                     context_observation_id=v.get('context_observation_id'),
                     evidence_observation_ids=v.get('evidence_observation_ids'))
                for v in self.task.progress_history[-10:]],
            history_scope='all_episode_factual_memory_projected_to_bounded_model_context')

    def read_memory_record(self, record_id):
        if record_id in self.keyframes:
            return dict(detail_requested=True,record=copy.deepcopy(self.keyframes[record_id]),
                sensor_record=copy.deepcopy(next(o for o in self.observations if o['observation_id']==record_id)),
                visual_analyses=copy.deepcopy([v for v in self.scene_history if v['source']['observation_id']==record_id]),
                grounding_ids=[g['grounding_id'] for g in self.groundings if g['source']['observation_id']==record_id])
        for values,key in [(self.tool_ledger,'call_id'),(self.groundings,'grounding_id'),
                           (self.visual_targets,'id')]:
            value=next((v for v in values if v[key]==record_id),None)
            if value is not None:return dict(detail_requested=True,record=copy.deepcopy(value))
        raise ValueError('unknown_memory_record')

    def stats(self):
        return dict(super().stats(),places=len(self.places),landmarks=len(self.landmarks),
            visual_targets=len(getattr(self,'visual_targets',[])),routes=len(self.routes))

    def snapshot(self):
        return dict(super().snapshot(),current_place_id=self.current_place_id,places=list(self.places.values()),
            keyframes=list(self.keyframes.values()),landmarks=list(self.landmarks.values()),routes=self._routes(),
            dynamic_records=copy.deepcopy(self.dynamic),dynamic=self.dynamic_summaries(),hypotheses=self.notes,
            visual_targets=copy.deepcopy(getattr(self,'visual_targets',[])),
            navigation_summary=self.navigation_summary())
