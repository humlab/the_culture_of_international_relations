import bokeh.palettes
import matplotlib
import holoviews as hv

default_palette = bokeh.palettes.Category20_20

def quantity_plot(
    data,
    pivot,
    chart_type,
    plot_style,
    overlay=True,
    figsize=(1000,600),
    xlabel='',
    ylabel='',
    xticks=None,
    yticks=None,
    xticklabels=None,
    yticklabels=None,
    xlim=None,
    ylim=None,
    dpi=48,
    colors=default_palette,
    **kwargs):

    matplotlib.style.use(plot_style)
    
    figsize=(figsize[0]/dpi, figsize[1]/dpi)
    
    kind = '{}{}'.format(chart_type.kind, 'h' if chart_type.horizontal else '')

    ax = pivot.plot(kind=kind, stacked=chart_type.stacked, figsize=figsize, color=colors, **kwargs)

    if xticks is not None:
        ax.set_xticks(xticks)
        
    if yticks is not None:
        ax.set_yticks(yticks)
        
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
        
    ax.set_xlabel(xlabel or '')
    ax.set_ylabel(ylabel or '')

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)
        
    box = ax.get_position()
    
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Put a legend to the right of the current axis
    legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    legend.get_frame().set_linewidth(0.0)

    if ax.get_xticklabels() is not None:
        for tick in ax.get_xticklabels():        
            tick.set_rotation(45)
# note: pd-read_csv can read .csv.gz
# from colorcet import fire
#  https://hvplot.pyviz.org/getting_started/index.html
#  http://pyviz.org/tutorial/01_Workflow_Introduction.html
#  measles_agg.loc[1980:1990, states].hvplot.bar('Year', by='State', rot=90)

def plot_treaties_per_period_holoviews(
    data,
    pivot,
    chart_type,
    plot_style,
    overlay=True,
    figsize=(1000,600),
    xlabel='',
    ylabel='',
    xticks=None,
    yticks=None,
    xticklabels=None,
    yticklabels=None,
    xlim=None,
    ylim=None,
    colors=bokeh.palettes.Category20[20]):
    
    plot_opts = dict(
        tools=['hover'],
        legend_position="right",
        xrotation=60,
        width=figsize[0],
        height=figsize[1],
        title_format="{label}",
        color_index='Party'
    )
    
    style_opts = dict(
        #cmap='Set1',
        box_color=hv.Cycle('Set1'), line_color=None, alpha=0.6
    )
    
    parties = list(data.Party.unique())
    periods = list(data.Period.unique())
    
    #if xticks is not None:
    #    plot_opts.update(dict(xticks=xticks))
        
    #if yticks is not None:
    #    plot_opts.update(dict(yticks=yticks))
    
    dataset = hv.Dataset(data)
    
    party_dim = hv.Dimension('Party', label='Party')
    period_dim = hv.Dimension('Period', label='Period')
    count_dim = hv.Dimension('Count', label='Count')
    
    if chart_type.kind == 'bar':

        group_type = '{}_index'.format('stack' if chart_type.stacked or len(parties) == 1 else 'group')
        
        plot_opts.update({ group_type: 'Party'})
        #plot_opts.update({ 'labelled': [None, 'y']})
        
        if overlay:
            p = dataset.to(hv.Bars, kdims=[period_dim, party_dim], vdims=count_dim, label='')\
                .redim.label(Party='X').opts(plot=plot_opts, style=style_opts)
            #p = hv.Bars(dataset, kdims=[period_dim, party_dim], vdims=count_dim, label='')\
            #    .opts(plot=plot_opts, style=style_opts)
            
        else:
            p = hv.Bars(data, kdims=[period_dim, party_dim], vdims=count_dim, label='')\
                .opts(plot=plot_opts, style=style_opts)
            
        display(p)
        
    if chart_type.kind == 'line':
        
        if xticks is not None:
            plot_opts.update(dict(xticks=xticks))        
            
        curves = {
            party: (hv.Curve(ds, kdims=period_dim, vdims=count_dim) * hv.Scatter(ds, kdims=period_dim, vdims=count_dim))
                for ds in (data[data.Party==party] for party in parties)
        }
        
        if overlay or len(curves.values()) == 1:
            p = hv.NdOverlay(curves).opts(plot=plot_opts).redim.range(x=periods)
        else:
            p = hv.Layout(list(curves.values())).cols(2)
            #.sort(['Period'])
        display(p)
        
def create_party_name_map(parties, palette=bokeh.palettes.Category20[20]):
    
    rev_dict = lambda d: {v: k for k, v in d.items()}

    df = parties.rename(columns=dict(short_name='party_short_name', country='party_country'))
    df['party'] = df.index

    rd = df[~df.group_no.isin([0, 8])][['party', 'party_short_name', 'party_name', 'party_country']].to_dict()

    party_name_map = {
        k: rev_dict(rd[k]) for k in rd.keys()
    }
    
    party_color_map = { party: palette[i % len(palette)] for i, party in enumerate(parties.index) }

    return party_name_map, party_color_map

def vstepper(vmax):
    if vmax < 15:
        return 1
    if vmax < 30:
        return 5
    if vmax < 500:
        return 10
    if vmax < 2000:
        return 100
    return 500
    
def prepare_plot_kwargs(data, chart_type, normalize_values, period_group):
    
    kwargs = dict(
        figsize=(1000, 500)
    )

    label = 'Number of treaties' if not normalize_values else 'Share%'

    c, v = ('x', 'y') if not chart_type.horizontal else ('y', 'x')
    kwargs.setdefault('%slabel' %(v,), label)
    
    if period_group['type'] == 'range':
        ticklabels = [ str(x) for x in period_group['periods'] ]
        
    else:
        ticklabels = [ '{} to {}'.format(x[0], x[1]) for x in period_group['periods'] ]
    vmax = data.max().max()

    vstep = vstepper(vmax)
    
    kwargs.setdefault('%sticks' %(c,), list(data.index))
    kwargs.setdefault('%sticks' %(v,), list(range(0,vmax+vstep, vstep)))
        
    if ticklabels is not None:
        kwargs.setdefault('%sticklabels' %(c,), ticklabels)

    if normalize_values:
        kwargs.setdefault('%slim' %(v,), [0, 100])
        
    return kwargs

