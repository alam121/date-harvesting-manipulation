"""Observation-only local corridor ranking. Capsule proxy, not collision certification."""
import math
import time
import numpy as np


def rank_corridors(goal, candidates, fruits, length=0.10, envelope_radius=0.045):
    goal = np.asarray(goal, dtype=float)
    if goal.shape != (3,) or not np.isfinite(goal).all():
        raise ValueError('invalid goal')
    distances = sorted((float(np.linalg.norm(np.asarray(f['base_xyz'])-goal)), i)
                       for i, f in enumerate(fruits))
    # Exempt exactly one confidently associated target; never erase all nearby dates.
    if not distances or distances[0][0] > 0.03:
        return {'status': 'TARGET_UNMATCHED', 'rows': []}
    if len(distances)>1 and distances[1][0]-distances[0][0] < 0.01:
        return {'status': 'TARGET_AMBIGUOUS', 'rows': []}
    target = distances[0][1]
    rows = []
    for candidate in candidates:
        direction = np.asarray(candidate['direction'], dtype=float)
        norm = float(np.linalg.norm(direction))
        if not math.isfinite(norm) or norm < 1e-8:
            continue
        end = np.asarray(candidate.get('end', goal), dtype=float)
        start = np.asarray(candidate.get('start', goal-direction/norm*length), dtype=float)
        segment = end-start
        if not np.isfinite(start).all() or not np.isfinite(end).all() or np.dot(segment,segment)<1e-12:
            raise ValueError('invalid insertion segment')
        nearby = []
        for i, fruit in enumerate(fruits):
            if i == target:
                continue
            center = np.asarray(fruit['base_xyz'], dtype=float)
            radius = float(fruit['radius_m'])
            if not np.isfinite(center).all() or not 0 < radius < .15:
                continue
            fraction = float(np.clip(np.dot(center-start,segment)/np.dot(segment,segment),0,1))
            clearance = float(np.linalg.norm(center-(start+fraction*segment))-radius-envelope_radius)
            if clearance < .03:
                nearby.append({'fruit_id': fruit['fruit_id'], 'clearance_m': clearance})
        interference = sum(max(0.,-f['clearance_m']) for f in nearby)
        rows.append({'label':candidate['label'], 'start':start.tolist(),'end':end.tolist(),
                     'interference_m':interference, 'neighbors':nearby,
                     'min_clearance_m':min((f['clearance_m'] for f in nearby),default=None)})
    rows.sort(key=lambda r:r['interference_m'])
    return {'status':'OK','target_id':fruits[target]['fruit_id'], 'rows':rows,
            'envelope_radius_m':envelope_radius,'length_m':length,
            'model':'capsule along supplied TCP segment; finger articulation not modeled'}


def publish_diagnostic(node, candidates, goal, length, snapshot_override=None):
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point
    import json
    snapshot = snapshot_override or getattr(node, '_nearby_dates_snapshot', None)
    result = {'status':'NO_OBSERVATIONS','rows':[]}
    if snapshot:
        received, payload = snapshot
        stamp = int(payload['stamp_ns'])
        source_age = (node.get_clock().now().nanoseconds-stamp)/1e9
        if snapshot_override is None and (time.monotonic()-received>1.0 or not 0<=source_age<=1.0):
            result['status']='STALE_OBSERVATIONS'
        else:
            result=rank_corridors(goal,candidates,payload['fruits'],length)
            result['source_age_s']=source_age
    node._last_neighbor_diagnostic = result
    node.get_logger().debug('[NEIGHBOR_DIAGNOSTIC] '+json.dumps(result))
    markers=MarkerArray()
    clear=Marker();clear.action=Marker.DELETEALL;markers.markers.append(clear)
    for i,row in enumerate(result['rows']):
        m=Marker();m.header.frame_id='base_link';m.header.stamp=node.get_clock().now().to_msg()
        m.ns='neighbor_corridors';m.id=i;m.type=Marker.LINE_STRIP;m.action=Marker.ADD
        m.pose.orientation.w=1.;m.scale.x=.005;m.color.a=.9
        m.color.r=1. if row['interference_m']>0 else 0.
        m.color.g=0.3 if row['interference_m']>0 else 1.
        m.lifetime.sec=10
        m.points=[Point(x=float(p[0]),y=float(p[1]),z=float(p[2])) for p in (row['start'],row['end'])]
        markers.markers.append(m)
        label=Marker();label.header=m.header;label.ns='neighbor_labels';label.id=i
        label.type=Marker.TEXT_VIEW_FACING;label.action=Marker.ADD;label.pose.orientation.w=1.
        label.pose.position=m.points[0];label.pose.position.z+=.015
        label.scale.z=.012;label.color=m.color;label.lifetime.sec=10
        label.text=f"{i+1}: {row['label']} ({row['interference_m']*1000:.0f}mm proxy overlap)"
        markers.markers.append(label)
    node._neighbor_marker_pub.publish(markers)


def collect_candidate(node, label, approach, final, goal):
    """Build candidates through production geometry before ranking. No motion/IK here."""
    import copy
    state = getattr(node, '_neighbor_geometry', None)
    if state is None:
        snapshot = getattr(node, '_nearby_dates_snapshot', None)
        if not snapshot:
            node._neighbor_geometry = {'disabled': True}
            return False
        received, payload = snapshot
        age = (node.get_clock().now().nanoseconds-int(payload['stamp_ns']))/1e9
        if time.monotonic()-received>1 or not 0<=age<=1:
            node._neighbor_geometry = {'disabled': True}
            node.get_logger().warn('[NEIGHBOR_RANK] stale observations; retaining axis ranking')
            return False
        state = {'snapshot':copy.deepcopy(snapshot), 'candidates':{}, 'labels':list(node._neighbor_candidate_labels)}
        node._neighbor_geometry = state
    if state.get('disabled') or state.get('done'):
        return False
    start, end = list(approach[:3]), list(final[:3])
    state['candidates'][label] = {'label':label,'start':start,'end':end,
                                'direction':(np.asarray(end)-start).tolist()}
    pending = [name for name in state['labels'] if name not in state['candidates']]
    if pending:
        node._corridor_forced_label = pending[0]
        return True
    publish_diagnostic(node,list(state['candidates'].values()),goal,.1,state['snapshot'])
    result = node._last_neighbor_diagnostic
    state['done'] = True
    node._corridor_forced_label = None
    if result['status']=='OK':
        node._neighbor_rank_order = [r['label'] for r in result['rows']]
        node.get_logger().debug('[NEIGHBOR_RANK] order='+str(node._neighbor_rank_order)+' geometry=PLANNED_TCP preflight=REQUIRED')
        return True  # reload preferred geometry and preflight it normally
    node.get_logger().warn('[NEIGHBOR_RANK] '+result['status']+'; retaining axis ranking')
    return True
