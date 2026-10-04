import unittest
from ur10e_curobo.neighbor_diagnostics import rank_corridors

class NeighborDiagnosticsTests(unittest.TestCase):
    def fruit(self, i, xyz):
        return dict(fruit_id=i,base_xyz=xyz,radius_m=.01)

    def test_local_interference_changes_ranking(self):
        candidates=[dict(label='straight',direction=[0,-1,0]),dict(label='side',direction=[1,0,0])]
        fruits=[self.fruit(1,[0,0,0]),self.fruit(2,[0,.08,0]),self.fruit(3,[10,10,10])]
        result=rank_corridors([0,0,0],candidates,fruits)
        self.assertEqual(result['rows'][0]['label'],'side')
        self.assertEqual(result['rows'][0]['interference_m'],0)
        self.assertGreater(result['rows'][1]['interference_m'],0)
        self.assertEqual([n['fruit_id'] for n in result['rows'][1]['neighbors']],[2])
        self.assertEqual(candidates[0]['label'],'straight')

    def test_ambiguous_target_is_unknown_not_clear(self):
        result=rank_corridors([0,0,0],[],[self.fruit(1,[0,0,0]),self.fruit(2,[.005,0,0])])
        self.assertEqual(result['status'],'TARGET_AMBIGUOUS')

    def test_missing_target(self):
        self.assertEqual(rank_corridors([0,0,0],[],[])['status'],'TARGET_UNMATCHED')

    def test_neighbor_beyond_path_does_not_block(self):
        result=rank_corridors([0,0,0],[dict(label='straight',direction=[0,-1,0])],
                             [self.fruit(1,[0,0,0]),self.fruit(2,[0,.3,0])])
        self.assertEqual(result['rows'][0]['neighbors'],[])

    def test_supplied_height_changes_clearance(self):
        fruits=[self.fruit(1,[0,0,0]),self.fruit(2,[0,.05,.055])]
        candidate=dict(label='level',direction=[0,-1,0],start=[0,.1,-.02],end=[0,0,-.02])
        result=rank_corridors([0,0,0],[candidate],fruits)
        self.assertEqual(result['rows'][0]['interference_m'],0)
        self.assertEqual(result['rows'][0]['end'],[0,0,-.02])
        candidate['start'][2]=.02;candidate['end'][2]=.02
        self.assertGreater(rank_corridors([0,0,0],[candidate],fruits)['rows'][0]['interference_m'],0)

    def test_collection_reloads_ranked_candidate(self):
        import time
        from types import SimpleNamespace as NS
        from unittest.mock import Mock, patch
        from ur10e_curobo.neighbor_diagnostics import collect_candidate
        node=NS(_nearby_dates_snapshot=(time.monotonic(),{'stamp_ns':10**9,'fruits':[]}),
                get_clock=lambda:NS(now=lambda:NS(nanoseconds=10**9)),
                _neighbor_candidate_labels=['a','b'],get_logger=lambda:Mock())
        self.assertTrue(collect_candidate(node,'a',[0,.1,0],[0,0,0],[0,0,0]))
        self.assertEqual(node._corridor_forced_label,'b')
        def publish(*args):
            node._last_neighbor_diagnostic={'status':'OK','rows':[{'label':'b'},{'label':'a'}]}
        with patch('ur10e_curobo.neighbor_diagnostics.publish_diagnostic',side_effect=publish):
            self.assertTrue(collect_candidate(node,'b',[.1,0,0],[0,0,0],[0,0,0]))
        self.assertEqual(node._neighbor_rank_order,['b','a'])
        self.assertIsNone(node._corridor_forced_label)
        self.assertFalse(collect_candidate(node,'b',[.1,0,0],[0,0,0],[0,0,0]))
