"""Trusted RGB-D, place graph, task and dynamic memory with bounded retrieval."""
import copy
import math
import time

from .contexts import fields
from .memory_records import MemoryRecords, task_tokens
from .motion import registered_action


def milestone_summary(value):
    proof=value.get('verification', {})
    return dict(subgoal=value['subgoal'],source_observation_ids=value.get('observation_ids',[])[-2:],
        executed_action_count=len(value.get('executed_steps',[])),
        record_id='task_%d' % value['subgoal'], evidence_state='task_verified',
        verification=fields(proof, ('completion_confirmed','supported','confirmed_phases','vertical_motion')))


class MemoryService(MemoryRecords):
    def __init__(self, runtime):
        self.runtime=runtime
        super().__init__(runtime.camera)

    def reset(self, episode_id, instruction):
        super().reset(episode_id,instruction)
        self.places,self.keyframes,self.landmarks,self.routes={},{},{},{}
        self.dynamic,self.notes=[],[]
        self.current_place_id=None
        self.place_visit_step=0

    def note_observation(self, record):
        pose=record['pose']
        previous=self.current_place_id
        place=self.places.get(previous)
        def separation(place):
            return math.sqrt((pose['x']-place['position'][0])**2+(pose['z']-place['position'][1])**2
                +(pose.get('y',0)-place['height_y_m_est'])**2)
        if place is None or separation(place) >= .5:
            near=[p for p in self.places.values() if p['motion_epoch']==pose.get('motion_epoch',0)
                and separation(p) < .5]
            self.current_place_id=near[0]['id'] if near else 'P%d' % len(self.places)
            if self.current_place_id not in self.places:
                self.places[self.current_place_id]=dict(id=self.current_place_id,
                    position=[pose['x'],pose['z']],height_y_m_est=pose.get('y',0), confidence='rgbd_estimate_not_global_localization',
                    position_uncertainty_m=pose['position_uncertainty'], motion_epoch=pose.get('motion_epoch',0),
                    observation_ids=[], headings_deg=[])
            place=self.places[self.current_place_id]
            if previous is not None and previous != self.current_place_id:
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

    def note_scene(self, scene, source):
        self.note_semantics(source,[v['name'] for v in scene.get('landmarks',[])])
        people=[dict(p,kind='person') for p in scene.get('humans',[])]+[
            dict(p,kind='uncertain_humanoid') for p in scene.get('uncertain_humanoids',[])]
        temporary=[dict(p,kind='temporary_obstacle') for p in scene.get('temporary_obstacles',[])]
        for item in people+temporary:
            self.dynamic.append(dict(id='dynamic_%04d' % len(self.dynamic),kind=item['kind'],
                place_id=self.keyframes[source['observation_id']]['place_id'],
                finding=item.get('activity',item.get('description','unknown activity')),
                box=item.get('box'),position=self.runtime.geometry().localize_human(item.get('box')),
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
            self.routes[path['id']]=dict(id=path['id'],waypoint_type=path['type'],from_place=self.current_place_id,to_place=None,
                state='CANDIDATE',traversability=path['first_step'],evidence_state='geometry_supported',
                source_observation_ids=[path['source_observation_id']],bearing_deg=path['bearing_deg'],
                requires_live_revalidation=True)

    def note_validation(self, path, value):
        route=self.routes.get(path['id'])
        if route and value['status']=='OK' and not value['executable_segment']['turn_only']:
            route.update(state='EXECUTABLE',evidence_state='geometry_supported',traversability='FREE',
                source_observation_ids=[self.current['observation_id']],validated_at=self.runtime.observation.captured_at)

    def note_motion(self):
        observations={o['observation_id']:o for o in self.observations}
        for route in self.routes.values():
            steps=route.get('motion_steps',[])
            if route['state']=='CANDIDATE' and route.get('to_place') and steps and all(
                    registered_action(self.actions[i],observations) for i in steps):
                route.update(state='TRAVERSED',evidence_state='motion_verified')

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

    def query_task_memory(self, subgoal_id=None):
        self.queries+=1
        task=self.task
        index=task.current_subgoal_index
        if subgoal_id is not None:
            if not subgoal_id.startswith('sg_') or not subgoal_id[3:].isdigit():
                raise ValueError('unknown_subgoal_id')
            index=int(subgoal_id[3:])
            if index>=len(task.subgoals):raise ValueError('unknown_subgoal_id')
        milestones=[m for m in task.completion_evidence if subgoal_id is None or m['subgoal']==index]
        feedback=self.progress_feedback or {}
        return dict(current_subgoal_id='sg_%d' % task.current_subgoal_index,
            goal=fields(task.subgoals[index],('action','target','landmark','success_criteria')) if index<len(task.subgoals) else None,
            goal_evidence_state='intent',verified_milestones=[milestone_summary(m) for m in milestones[-4:]],
            missing_evidence=feedback.get('missing',[]) if feedback.get('subgoal')==index else ['fresh_task_verification'],
            hypotheses=copy.deepcopy(self.notes[-2:]))

    def read_memory_record(self, record_id):
        if record_id in self.keyframes:
            return dict(detail_requested=True,record=copy.deepcopy(self.keyframes[record_id]),
                sensor_record=copy.deepcopy(next(o for o in self.observations if o['observation_id']==record_id)),
                visual_analyses=copy.deepcopy([v for v in self.scene_history if v['source']['observation_id']==record_id]),
                grounding_ids=[g['grounding_id'] for g in self.groundings if g['source']['observation_id']==record_id])
        for values,key in [(self.tool_ledger,'call_id'),(self.groundings,'grounding_id')]:
            value=next((v for v in values if v[key]==record_id),None)
            if value is not None:return dict(detail_requested=True,record=copy.deepcopy(value))
        if record_id.startswith('task_') and record_id[5:].isdigit():
            value=next((v for v in self.task.completion_evidence if v['subgoal']==int(record_id[5:])),None)
            if value is not None:return dict(detail_requested=True,record=copy.deepcopy(value))
        raise ValueError('unknown_memory_record')

    def stats(self):
        return dict(super().stats(),places=len(self.places),landmarks=len(self.landmarks),routes=len(self.routes))

    def snapshot(self):
        return dict(super().snapshot(),current_place_id=self.current_place_id,places=list(self.places.values()),
            keyframes=list(self.keyframes.values()),landmarks=list(self.landmarks.values()),routes=self._routes(),
            dynamic_records=copy.deepcopy(self.dynamic),dynamic=self.dynamic_summaries(),hypotheses=self.notes)
