
# coding: utf-8

# In[101]:

import numpy as np
from numpy.random import randn
import matplotlib.mlab as mlab
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from numpy.polynomial import Chebyshev as T


# In[45]:

# line plot
def linePlot(save=False):
    x = np.linspace(0,10,500)
    y = np.sin(x)*x
    plt.plot(x,y,linewidth=3)
    if(save==False):
        plt.show()
    else:
        plt.title("line plot")
        plt.savefig("line_plot.png")
        plt.close()


# In[46]:

# log plot
def logPlot(save=False):
    x = np.linspace(10,1000)
    plt.semilogy(x,x**5,'g')
    plt.semilogy(x,x**4,'r')
    if(save==False):
        plt.show()
    else:
        plt.title("log plot")
        plt.savefig("log_plot.png")
        plt.close()


# In[47]:

# lin-lin plot
def linlinPlot(save=False):
    x = np.linspace(10,1000)
    plt.plot(x,x**5,'g',lw=3)
    if(save==False):
        plt.show()
    else:
        plt.title("lin-lin plot")
        plt.savefig("lin_lin_plot.png")
        plt.close()


# In[48]:

# bar graph
def barGraph(save=False):
    data = {'Spam':22, 'Eggs':21, 'Ham':20, 'Sausage':19, 'Bacon':18, 'Baked beans':10,'Lobster thermidor':11}
    val = list(data.itervalues()) # the bar lengths
    labels = list(data.iterkeys()) # the labels for each bar
    pos = np.arange(7)+.5    # the bar centers on the y axis
    plt.figure(1)
    plt.bar(pos,val,align='center')
    plt.xticks(pos,labels[::-1],rotation='vertical')
    plt.yticks([])
    if(save==False):
        plt.show()
    else:
        plt.title("bar graph")
        plt.savefig("bar_graph.png")
        plt.close()


# In[49]:

# dot plot TODO
def dotPlot(save=False):
    data = {'Spam':22, 'Eggs':21, 'Ham':20, 'Sausage':19, 'Bacon':18, 'Baked beans':10,'Lobster thermidor':11}
    val = list(data.itervalues()) # the bar lengths
    labels = list(data.iterkeys()) # the labels for each bar
    pos = np.arange(7)+.5    # the bar centers on the y axis
    plt.figure(1)
    plt.plot(pos,val, align='center')
    plt.xticks(pos, labels[::-1],rotation='vertical')
    plt.yticks([])
    if(save==False):
        plt.show()
    else:
        plt.title("dot plot")
        plt.savefig("dot_plot.png")
        plt.close()


# In[50]:

# histogram
def histogram(save=False):
    data = randn(75)+randn(75)+randn(75) + randn(75)
    plt.hist(data)
    if(save==False):
        plt.show()
    else:
        plt.title("histogram")
        plt.savefig("histogram.png")
        plt.close()


# In[53]:

# histogram with best fit line
def histogramFancy(save=False):
    mu, sigma = 100, 15
    x = mu + sigma*np.random.randn(10000)

    # the histogram of the data
    n, bins, patches = plt.hist(x, 50, normed=1, edgecolor='none',facecolor='green',alpha=.5)

    # add a 'best fit' line
    y = mlab.normpdf( bins, mu, sigma)
    #l = plt.plot(bins, y, 'r--', linewidth=1)

    #plt.xlabel('Smarts')
    #plt.ylabel('Probability')
    #plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
    #plt.axis([40, 160, 0, 0.03])
    plt.grid(True, color='w',linestyle='-')
    if(save==False):
        plt.show()
    else:
        plt.title("histogram with best fit line")
        plt.savefig("histogram_fancy.png")
        plt.close()
        


# In[55]:

# scatter plot
def scatterPlot(save=False):
    x = np.random.rand(100)
    y = 2*np.random.rand(100) + x - 1
    plt.scatter(x,y)
    plt.xlim(0,1)
    plt.ylim(-1,2)
    if(save==False):
        plt.show()
    else:
        plt.title("scatter plot")
        plt.savefig("scatter_plot.png")
        plt.close()


# In[58]:

# dual scatter plot
def dualscatterPlot(save=False):
    x1 = np.random.rand(30)
    x2 = np.random.rand(30)
    y1 = 2*np.random.rand(30) + x1 - 1
    y2 = 2*np.random.rand(30) - x2 - 1
    plt.scatter(x1,y1,color='r')
    plt.scatter(x2,y2,color='g')
    plt.xlim(0,1)
    plt.ylim(-1,2)
    if(save==False):
        plt.show()
    else:
        plt.title("dual scatter plot")
        plt.savefig("scatter_dual_plot.png")
        plt.close()


# In[60]:

# contour map basic
def contourMap(save=False):
    n=400
    xrange = np.linspace(-1.5,1.5,n)
    yrange = np.linspace(-1.5,1.5,n)
    X, Y = np.meshgrid(xrange,yrange)
    F = Y**2 - X**3 + X**2
    plt.contour(X, Y, F, [-2,-1,0.0001,1,2,3,4,5] ,linewidths=3.0,cmap=plt.get_cmap('Blues'))
    if(save==False):
        plt.show()
    else:
        plt.title("contour map")
        plt.savefig("contour_map.png")
        plt.close()


# In[62]:

# filled contour map
def contourMap_filled(save=False):
    n=400
    xrange = np.linspace(-1.5,1.5,n)
    yrange = np.linspace(-1.5,1.5,n)
    X, Y = np.meshgrid(xrange,yrange)
    F = Y**2 - X**3 + X**2
    plt.contourf(X, Y, F, [-2,-1,0.0001,1,2,3,4,5] ,cmap=plt.get_cmap('afmhot'))
    if(save==False):
        plt.show()
    else:
        plt.title("filled contour map")
        plt.savefig("contour_map_filled.png")
        plt.close()


# In[68]:

# rainbow filled contour map
def contourMap_rainbowFilled(save=False):
    n=400
    xrange = np.linspace(-1.5,1.5,n)
    yrange = np.linspace(-1.5,1.5,n)
    X, Y = np.meshgrid(xrange,yrange)
    F = Y**2 - X**3 + X**2
    plt.contourf(X, Y, F, [-2,-1,0.0001,1,2,3,4,5])
    if(save==False):
        plt.show()
    else:
        plt.title("rainbow filled contour map")
        plt.savefig("contour_map_rainbow_filled.png")
        plt.close()


# In[66]:

# rainbow outline contour map
def contourMap_rainbow(save=False):
    n=400
    xrange = np.linspace(-1.5,1.5,n)
    yrange = np.linspace(-1.5,1.5,n)
    X, Y = np.meshgrid(xrange,yrange)
    F = Y**2 - X**3 + X**2
    plt.contour(X, Y, F, [-2,-1,0.0001,1,2,3,4,5] ,linewidths=3.0)
    if(save==False):
        plt.show()
    else:
        plt.title("rainbow contour map")
        plt.savefig("contour_map_rainbow.png")
        plt.close()


# In[71]:

# hue
def hue(save=False):
    from matplotlib.colors import from_levels_and_colors
    cmap, norm = from_levels_and_colors(range(8), ['black','darkviolet','blue','green', 'yellow','orange', 'red'])

    plt.figure(figsize=(10, 0.5))
    np.random.seed(10)
    plt.pcolormesh(np.arange(7).reshape(1, -1),cmap=cmap, norm=norm) 
    ax = plt.gca() 
    ax.yaxis.set_visible(False) 
    ax.xaxis.set_visible(False) 
    ax.autoscale_view(tight=True)
    if(save==False):
        plt.show()
    else:
        plt.title("hue")
        plt.savefig("hue.png")
        plt.close()


# In[74]:

# color value
def colorValue(save=False):
    plt.figure(figsize=(10, 0.5))
    plt.pcolormesh(np.arange(8).reshape(1, -1),cmap='Greens') 
    ax = plt.gca() 
    ax.yaxis.set_visible(False) 
    ax.xaxis.set_visible(False) 
    ax.autoscale_view(tight=True)
    if(save==False):
        plt.show()
    else:
        plt.title("color value")
        plt.savefig("color_value.png")
        plt.close()


# In[76]:

# example of a color map
def colorMap(save=False):
    plt.figure(figsize=(10, 0.5))
    plt.pcolormesh(np.arange(8).reshape(1, -1),cmap='afmhot') 
    ax = plt.gca() 
    ax.yaxis.set_visible(False) 
    ax.xaxis.set_visible(False) 
    ax.autoscale_view(tight=True)
    if(save==False):
        plt.show()
    else:
        plt.title("color map")
        plt.savefig("color_map.png")
        plt.close()


# In[78]:

# perception error: varying bases
def PE_varyingBases(save=False):
    x = np.linspace(0,10, 500)
    y = x**2
    plt.plot(x,y)
    plt.plot(x,y+20)
    fig = plt.gcf()
    fig.set_size_inches(5,7)
    if(save==False):
        plt.show()
    else:
        plt.title("perception error: varying bases")
        plt.savefig("PE_varying_bases.png")
        plt.close()
PE_varyingBases(save=True)


# In[80]:

# perception error: problematic area
def PE_problematic_area(save=False):
    circle1=plt.Circle((0.2,0.2),.1,color='r')
    circle2=plt.Circle((.2,.6),.1*np.sqrt(2),color='b')
    circle3=plt.Circle((.6,.3),.2,color='g',clip_on=False)
    fig = plt.gcf()
    fig.gca().set_aspect('equal')
    fig.gca().add_artist(circle1)
    #fig.gca().add_artist(circle2)
    fig.gca().add_artist(circle3)
    plt.axis('off')

    if(save==False):
        plt.show()
    else:
        plt.title("perception error: problematic area")
        plt.savefig("PE_problematic_area.png")
        plt.close()


# In[83]:

# pie chart
def pieChart(save=False):
    # make a square figure and axes
    plt.figure(1, figsize=(6,6))
    ax = plt.axes([0.1, 0.1, 0.8, 0.8])

    # The slices will be ordered and plotted counter-clockwise.
    labels = 'Spam', 'Eggs', 'Ham', 'Sausage', 'Bacon', 'Baked beans','Lobster thermidor'
    fracs = range(22,15,-1)
    fracs[-1]=10
    fracs[-2]=11

    data = zip(labels,fracs)
    #print data
    ##Rearrange the slices and labels
    r = [0,6,1,4,2,5,3]
    rfracs = [fracs[x] for x in r]
    rlabels = [labels[x] for x in r]
    ##end rearrange
    #explode=(0, 0, 0, 0)
    wedges, texts = plt.pie(rfracs,labels=rlabels)
    #pie(fracs, explode=explode, labels=labels,
    #                autopct='%1.1f%%', shadow=True, startangle=90)
                    # The default startangle is 0, which would start
                    # the Frogs slice on the x-axis.  With startangle=90,
                    # everything is rotated counter-clockwise by 90 degrees,
                    # so the plotting starts on the positive y-axis.


    for w in wedges:
        w.set_linewidth( 0 )
        w.set_edgecolor( 'none' )
    plt.title("Which is greater: lobster or beans?")
    if(save==False):
        plt.show()
    else:
        #plt.title("pie chart")
        plt.savefig("pie_chart.png")
        plt.close()


# In[85]:

# pie chart sorted
def pieChart_sorted(save=False):
    # make a square figure and axes
    plt.figure(1, figsize=(6,6))
    ax = plt.axes([0.1, 0.1, 0.8, 0.8])

    # The slices will be ordered and plotted counter-clockwise.
    labels = 'Spam', 'Eggs', 'Ham', 'Sausage', 'Bacon', 'Baked beans','Lobster thermidor'
    fracs = range(22,15,-1)
    fracs[-1]=10
    fracs[-2]=11

    data = zip(labels,fracs)
    #print data
    ##Rearrange the slices and labels
    r = [0,6,1,4,2,5,3]
    rfracs = [fracs[x] for x in r]
    rlabels = [labels[x] for x in r]
    ##end rearrange
    
    wedges, texts = plt.pie(fracs,labels=labels)
    #pie(fracs, explode=explode, labels=labels,
    #                autopct='%1.1f%%', shadow=True, startangle=90)
                    # The default startangle is 0, which would start
                    # the Frogs slice on the x-axis.  With startangle=90,
                    # everything is rotated counter-clockwise by 90 degrees,
                    # so the plotting starts on the positive y-axis.

    for w in wedges:
        w.set_linewidth( 0 )
        w.set_edgecolor( 'none' )
                
    plt.title('Which is greater: lobster or beans?')

    if(save==False):
        plt.show()
    else:
        #plt.title("pie chart sorted")
        plt.savefig("pie_chart_sorted.png")
        plt.close()


# In[87]:

# bar chart
def barChart(save=False):
    # The slices will be ordered and plotted counter-clockwise.
    labels = 'Spam', 'Eggs', 'Ham', 'Sausage', 'Bacon', 'Baked beans','Lobster thermidor'
    fracs = range(22,15,-1)
    fracs[-1]=10
    fracs[-2]=11
    
    val = fracs[::-1]    # the bar lengths
    pos = np.arange(7)+.5    # the bar centers on the y axis

    plt.figure(1)
    plt.barh(pos,val, align='center')
    plt.yticks(pos, labels[::-1])
    plt.xticks([])
    plt.title('More lobster or beans?')
    plt.grid(False)
    if(save==False):
        plt.show()
    else:
        plt.title("bar chart")
        plt.savefig("bar_chart.png")
        plt.close()


# In[90]:

# bar chart unsorted
def barChart_unsorted(save=False):
    # The slices will be ordered and plotted counter-clockwise.
    labels = 'Spam', 'Eggs', 'Ham', 'Sausage', 'Bacon', 'Baked beans','Lobster thermidor'
    fracs = range(22,15,-1)
    fracs[-1]=10
    fracs[-2]=11

    data = zip(labels,fracs)
    #print data
    ##Rearrange the slices and labels
    r = [0,6,1,4,2,5,3]
    rfracs = [fracs[x] for x in r]
    rlabels = [labels[x] for x in r]
    ##end rearrange
    
    plt.figure(1)
    plt.barh(pos,rfracs, align='center')
    plt.yticks(pos, rlabels)
    plt.xticks([])

    plt.title('Unsorted spam, lobster, and beans')
    plt.grid(False)
    
    
    if(save==False):
        plt.show()
    else:
        #plt.title("")
        plt.savefig("bar_chart_unsorted.png")
        plt.close()
        


# In[94]:

# bar chart vertical
def barChart_verticalBars(save=False):
    # The slices will be ordered and plotted counter-clockwise.
    labels = 'Spam', 'Eggs', 'Ham', 'Sausage', 'Bacon', 'Baked beans','Lobster thermidor'
    fracs = range(22,15,-1)
    fracs[-1]=10
    fracs[-2]=11
    
    val = fracs[::-1]    # the bar lengths
    pos = np.arange(7)+.5    # the bar centers on the y axis
    plt.figure(1)
    plt.bar(pos,val, align='center')
    plt.xticks(pos, labels[::-1])
    plt.yticks([])
    #plt.title('spam, lobster, and beans')

    plt.grid(False)
    
    if(save==False):
        plt.show()
    else:
        plt.title("vertical bar chart")
        plt.savefig("bar_chart_vertical_bars.png")
        plt.close()


# In[96]:

# bar chart with vertical labels
def barChart_verticalLabels(save=False):
    # The slices will be ordered and plotted counter-clockwise.
    labels = 'Spam', 'Eggs', 'Ham', 'Sausage', 'Bacon', 'Baked beans','Lobster thermidor'
    fracs = range(22,15,-1)
    fracs[-1]=10
    fracs[-2]=11
    
    val = fracs[::-1]    # the bar lengths
    pos = np.arange(7)+.5    # the bar centers on the y axis
    
    plt.figure(1)
    plt.bar(pos,val, align='center')
    plt.xticks(pos, labels[::-1],rotation='vertical')
    plt.yticks([])
    #plt.title('spam, lobster, and beans')

    plt.grid(False)
    if(save==False):
        plt.show()
    else:
        plt.title("bar chart with vertical labels")
        plt.savefig("bar_chart_vertical_labels.png")
        plt.close()


# In[100]:

# lots of small multiple images
def multiples(save=False):
    t = np.arange(0.0, 1.0, 0.1)
    s = np.sin(2*np.pi*t)
    linestyles = ['_', '-', '--', ':']
    markers = []
    for m in Line2D.markers:
        try:
            if len(m) == 1 and m != ' ':
                markers.append(m)
        except TypeError:
            pass

    styles = markers + [
        r'$\lambda$',
        r'$\bowtie$',
        r'$\Gamma$',
        r'$\clubsuit$',
        r'$\checkmark$']

    colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

    plt.figure(figsize=(8,8))

    axisNum = 0
    for row in range(6):
        for col in range(5):
            axisNum += 1
            ax = plt.subplot(6, 5, axisNum)
            color = colors[axisNum % len(colors)]
            if axisNum < len(linestyles):
                plt.plot(t, s, linestyles[axisNum], color=color, markersize=5)
            else:
                style = styles[(axisNum - len(linestyles)) % len(styles)]
                plt.plot(t, s, linestyle='None', marker=style, color=color, markersize=5)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.yaxis.set_ticks_position('none')
            ax.xaxis.set_ticks_position('none')
    if(save==False):
        plt.show()
    else:
        #plt.title("multiples")
        plt.savefig("multiples.png")
        plt.close()


# In[103]:

# plotting on top
def PoT(save=False):
    x = np.linspace(-1,1,500)
    for i in range(6):
       plt.plot(x, T.basis(i)(x), lw=2, label="$T_%d$"%i)
    plt.legend(loc="upper left")
    
    if(save==False):
        plt.show()
    else:
        plt.title("plotting Chebyshev polynomials on top of each other")
        plt.savefig("PoT.png")
        plt.close()


# In[108]:

# plotting on top: extend boundary
def PoT_boundary(save=False):
    x = np.linspace(-1,1,500)
    plt.xlim(-1.1,1.1)
    plt.ylim(-1.1,1.1)

    for i in range(6):
       plt.plot(x, T.basis(i)(x), lw=2, label="$T_%d$"%i)

    plt.legend(loc="upper left")
    if(save==False):
        plt.show()
    else:
        plt.title("extend boundary")
        plt.savefig("PoT_extend_boundary.png")
        plt.close()


# In[109]:

# plotting on top: move legend
def PoT_legend(save=False):
    x = np.linspace(-1,1,500)
    plt.xlim(-1.1,1.1)
    plt.ylim(-1.1,1.1)

    for i in range(6):
       plt.plot(x, T.basis(i)(x), lw=2, label="$T_%d$"%i)

    ax = plt.gca()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if(save==False):
        plt.show()
    else:
        plt.title("legend to the right")
        plt.savefig("PoT_legend.png")
        plt.close()


# In[111]:

# plotting separately
def PoT_separate(save=False):
    fig = plt.figure(dpi=100)
    fig.set_size_inches(10,10)
    fig.suptitle('Chebyshev Polynomials', fontsize=20)

    x = np.linspace(-1,1,500)

    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        ax.plot(x, T.basis(i)(x), lw=2, label="$T_%d$"%i)
        plt.xlim(-1.1,1.1)
        plt.ylim(-1.1,1.1)
        ax.set_title('$T_%d$'%i)
        if i%3:
            ax.set_yticklabels([])
            ax.yaxis.set_ticks_position('none')
        if i<6:
            ax.xaxis.set_ticks_position('none')
            ax.set_xticklabels([]) 
    if(save==False):
        plt.show()
    else:
        plt.title("separate plots")
        plt.savefig("PoT_separate.png")
        plt.close()


# In[113]:

# scale matters: scatter
def scale_scatter(save=False):
    x = np.random.rand(100)
    y = 2*np.random.rand(100) + x - 1
    plt.scatter(x,y)
    plt.xlim(0,1)
    plt.ylim(-1,2)
    if(save==False):
        plt.show()
    else:
        plt.title("zoomed-in scale")
        plt.savefig("scale_scatter_zoomed_in.png")
        plt.close()
    n=4
    plt.scatter(x,y)
    plt.xlim(-.1,1.1)
    plt.ylim(-n,n+1)
    if(save==False):
        plt.show()
    else:
        plt.title("zoomed-out scale")
        plt.savefig("scale_scatter_zoomed_out.png")
        plt.close()


# In[116]:

# scale matters: line
def scale_line(save=False):
    x = np.linspace(0,1,200)
    y = 100+10*np.random.rand(200)
    plt.plot(x,y)
    plt.xlim(-.1,1.1)

    if(save==False):
        plt.show()
    else:
        plt.title("zoomed-in scale")
        plt.savefig("scale_line_zoomed_in.png")
        plt.close()
    
    plt.plot(x,y)
    plt.xlim(-.1,1.1)
    plt.ylim(0,155)
    
    if(save==False):
        plt.show()
    else:
        plt.title("zoomed-out scale")
        plt.savefig("scale_line_zoomed_out.png")
        plt.close()


# In[118]:

# window location
def windowLoc(save=False):
    x = np.linspace(0,10,500)
    y = np.exp(-x)*x
    plt.plot(x,y)
    plt.xlim(0,.5)
    plt.ylim(0,.35)
    if(save==False):
        plt.show()
    else:
        plt.title("window location 1")
        plt.savefig("window_location_1.png")
        plt.close()
    
    plt.plot(x,y)
    plt.xlim(2,4)
    if(save==False):
        plt.show()
    else:
        plt.title("window location 2")
        plt.savefig("window_location_2.png")
        plt.close()
    
    plt.plot(x,y)
    plt.xlim(0,10)
    if(save==False):
        plt.show()
    else:
        plt.title("window location 3")
        plt.savefig("window_location_3.png")
        plt.close()


# In[119]:

if __name__ == "__main__":

    linePlot(save=True)
    logPlot(save=True)
    linlinPlot(save=True)
    barGraph(save=True)
    #dotPlot(save=False)
    histogram(save=True)
    histogramFancy(save=True)
    scatterPlot(save=True)
    dualscatterPlot(save=True)
    contourMap(save=True)
    contourMap_filled(save=True)
    contourMap_rainbowFilled(save=True)
    contourMap_rainbow(save=True)
    hue(save=True)
    colorValue(save=True)
    colorMap(save=True)
    PE_problematic_area(save=True)
    pieChart(save=True)
    pieChart_sorted(save=True)
    barChart(save=True)
    barChart_unsorted(save=True)
    barChart_verticalBars(save=True)
    barChart_verticalLabels(save=True)
    multiples(save=True)
    PoT(save=True)
    PoT_boundary(save=True)
    PoT_legend(save=True)
    PoT_separate(save=True)
    scale_scatter(save=True)
    scale_line(save=True)
    windowLoc(save=True)
    


# In[ ]:



