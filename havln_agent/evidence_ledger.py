"""Sensor evidence: measured pre-forward obstacles and no-change motion stalls.

A repeated source observation ID is not a physical location key. We group
byte-identical RGB-D at the same commanded heading/pitch and local pose
epoch even when a refresh assigned a different observation ID.
"""
import copy
import math


def physical_view_key(record):
    """Conservative exact-view grouping; not privileged/global localization."""
    rgb,depth=record.get('rgb_hash'),record.get('depth_hash')
    pixels=(rgb,depth) if rgb and depth else (
        'unhashed_observation',record.get('observation_id'))
    position=record.get('local_position_xz_m_est') or (None,None)
    return (pixels,round(float(record.get('heading_deg') or 0),1),
            round(float(record.get('pitch_deg') or 0),1),
            record.get('motion_epoch'),record.get('viewpoint_generation'),
            tuple(round(float(v),2) if v is not None else None
                  for v in position))


class EvidenceLedger:
    def __init__(self):
        self.obstacle_blocks=[]
        self.motion_stalls=[]
        self.surface_measurements=[]

    @staticmethod
    def _source_view(source, pose):
        return dict(
            observation_id=source['observation_id'],
            rgb_hash=source.get('rgb_hash'),
            depth_hash=source.get('depth_hash'),
            heading_deg=round(math.degrees(pose['heading'])%360,2),
            pitch_deg=pose.get('pitch_deg'),
            local_position_xz_m_est=[pose.get('x'),pose.get('z')],
            motion_epoch=pose.get('motion_epoch'),
            viewpoint_generation=pose.get('viewpoint_generation'))

    def blocked_forward(self, source, pose, *, reason, center, place_id=None):
        """Record an atomic move veto; no simulator MOVE was executed."""
        item=dict(self._source_view(source,pose),
            place_id=place_id,reason=reason,
            central_support_pixels=center.get('central_support_pixels'),
            nearest_along_m=center.get('nearest_along_m'),
            focus_region_xyxy_norm=center.get('focus_region_xyxy_norm'),
            source_pixel_xy=center.get('source_pixel_xy'),
            forward_atomic_action_executed=False,
            evidence_scope='one_current_stance_and_bearing_not_global_route_veto')
        key=physical_view_key(item)
        repeat=1+sum(physical_view_key(v)==key
                     for v in self.obstacle_blocks)
        item['repeat_count_at_recording']=repeat
        self.obstacle_blocks.append(item)
        return copy.deepcopy(item)

    def stalled_forward(self, source, pose, *, action_step, after_observation_id):
        """Actual MOVE receipt but same RGB and Depth afterwards.

        This is NOT a measured wall and not proof of physical collision.
        """
        item=dict(self._source_view(source,pose),
            action_step=action_step,
            after_observation_id=after_observation_id,
            type='NO_OBSERVED_FRAME_CHANGE',
            forward_atomic_action_executed=True,
            collision_proven=False,
            displacement_verified=False)
        key=physical_view_key(item)
        item['repeat_count_at_recording']=1+sum(
            physical_view_key(v)==key for v in self.motion_stalls)
        self.motion_stalls.append(item)
        return copy.deepcopy(item)

    def note_surface(self, measurement):
        if measurement.get('status') not in ('MEASURED','INSUFFICIENT_DEPTH'):
            raise ValueError('invalid_surface_evidence')
        self.surface_measurements.append(copy.deepcopy(measurement))
        return copy.deepcopy(measurement)

    def summary(self):
        return dict(
            latest_obstacle_blocks=copy.deepcopy(self.obstacle_blocks[-4:]),
            latest_motion_stalls=copy.deepcopy(self.motion_stalls[-4:]),
            latest_surface_measurements=copy.deepcopy(self.surface_measurements[-3:]))
