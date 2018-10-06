
from common.network.graphtool_utility import nx2gt
from types import SimpleNamespace as bunch

try:
    import graph_tool.draw as gt_draw
    import graph_tool.all as gt
except:
    print('warning: graph_tool not installed!')
    
def sfdp_args(G, **kwargs):
    weight = G_gt.edge_properties['weight']
    K = kwargs.get('K', 0.1)
    C = kwargs.get('C', 1.0)
    p = kwargs.get('p', 0.1)
    return dict(eweight=weight, K=K, C=C/100.0, gamma=p)
        
def arf_args(G, **kwargs):
    weight = G_gt.edge_properties['weight']
    K = kwargs.get('K', 0.1)
    C = kwargs.get('C', 1.0)
    return dict(weight=weight, d=K, a=C)
        
def fruchterman_reingold_args(G, **kwargs):
    weight = G_gt.edge_properties['weight']
    K = kwargs.get('K', 0.1)
    C = kwargs.get('C', 1.0)
    N = len(G)
    return dict(weight=weight, a=(2.0*N*K), r=2.0*C)

def layout_network(G, **kwargs):
    
    global layout_setup_map
    
    setup = layout_setup_map[kwargs['layout_algorithm']]
    
    G_gt = nx2gt(G)
    G_gt.set_directed(False)
    
    args = setup.layout_args(G, **kwargs)
    
    layout_gt = setup.layout_function(G_gt, **args)
    
    layout = { G_gt.vertex_properties['id'][i]: layout_gt[i] for i in G_gt.vertices() }
    
    return layout, (G_gt, layout_gt)

layout_setups = [
    
    bunch(key='graphtool_arf',
          package='graphtool',
          name='graph-tool (arf)',
          layout_network=layout_network,
          layout_function=gt_draw.arf_layout,
          layout_args=arf_args),
    
    bunch(key='graphtool_sfdp',
          package='graphtool',
          name='graph-tool (sfdp)',
          layout_network=layout_network,
          layout_function=gt_draw.sfdp_layout,
          layout_args=sfdp_args),
    
    bunch(key='graphtool_fr',
          package='graphtool',
          name='graph-tool (FR)',
          layout_network=layout_network,
          layout_function=gt_draw.fruchterman_reingold_layout,
          layout_args=fruchterman_reingold_args)
]

layout_setup_map = { x.key: x for x in layout_setups }
