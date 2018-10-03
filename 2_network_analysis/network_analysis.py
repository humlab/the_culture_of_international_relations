

def network_edges_to_dicts(network, layout):
    LD = [ extend(dict(source=u,target=v,xs=[layout[u][0], layout[v][0]], ys=[layout[u][1], layout[v][1]]), d) for u, v, d in G.edges(data=True) ]
    LD.sort(key=lambda x: x['signed'])
    edges = dict(zip(LD[0],zip(*[d.values() for d in LD])))
    return edges

def pandas_to_network_edges(data):
    return [ (x[0], x[1], { y: x[j] for j, y in enumerate(data.columns)}) for i, x in data.iterrows() ]
    
def get_party_network_data(state=None, period_group=None, party_name='party_name', parties=None, treaty_filter='', extra_category='', n_top=0):

    period_column = period_group['column']
    
    treaty_subset = state.get_treaties_within_division(state.treaties, period_group, treaty_filter)
    xxx = state.stacked_treaties.loc[treaty_subset.index] #  state.stacked_treaties[state.stacked_treaties.index.isin(treaty_subset.index)]

    data = state.stacked_treaties.copy()

    data = data.loc[(data.signed_period!='other')]

    if only_is_cultural:
        data = data.loc[(data.is_cultural==True)]

    if isinstance(parties, list):
        data = data.loc[(data.party.isin(parties))]
    else:
        data = data.loc[(data.reversed==False)]

    data = data.loc[(data.signed_period != period)]
    data = data.loc[(data.signed_year.between(period[0], period[1]))]
    data = data.sort_values('signed')

    # data = data.groupby(['party', 'party_other']).size().reset_index().rename(columns={0: 'weight'})
    
    data = data[[ 'party', 'party_other', 'signed', 'topic', 'headnote']]

    if party_name != 'party':
        for column in ['party', 'party_other']:
            data[column] = data[column].apply(lambda x: state.get_party_name(x, party_name))

    data['weight'] = 1.0
            
    return data

def create_party_network(data, K, node_partition, palette): #, multigraph=True):

    #if multigraph:
    
    edges_data = pandas_to_network_edges(data)

    G = nx.MultiGraph(K=K)
    G.add_edges_from(edges_data)
    #else:
    #    edges_data = [ tuple(x) for x in data.values ]
    #    print(edges_data)
    #    G = nx.Graph(K=K)
    #    G.add_weighted_edges_from(edges_data)

    if node_partition is not None:
        partition = community.best_partition(G)
        partition_color = { n: palette[p % len(palette)] for n, p in partition.items() }
        nx.set_node_attributes(G, partition, 'community')
        nx.set_node_attributes(G, partition_color, 'fill_color')
    else:
        #nx.set_node_attributes(G, 0, 'community')
        nx.set_node_attributes(G, palette[0], 'fill_color')

    nx.set_node_attributes(G, dict(G.degree()), name='degree')
    nx.set_node_attributes(G, dict(nx.betweenness_centrality(G, weight=None)), name='betweenness')
    nx.set_node_attributes(G, dict(nx.closeness_centrality(G)), name='closeness')
    
    # if not multigraph:
    #    nx.set_node_attributes(G, dict(nx.eigenvector_centrality(G, weight=None)), name='eigenvector')
        
    # nx.set_node_attributes(G, dict(nx.communicability_betweenness_centrality(G)), name='communicability_betweenness')
    
    return G

def setup_node_size(nodes, node_size, node_size_range):

    if node_size in nodes.keys() and node_size_range is not None:
        nodes['clamped_size'] = clamp_values(nodes[node_size], node_size_range)
        node_size = 'clamped_size'
    return node_size
    
def setup_label_y_offset(nodes, node_size):

    label_y_offset = 'y_offset' if node_size in nodes.keys() else node_size + 5
    if label_y_offset == 'y_offset':
        nodes['y_offset'] = [ y + r for (y, r) in zip(nodes['y'], [ r / 2.0 + 5 for r in nodes[node_size] ]) ]
    return label_y_offset
