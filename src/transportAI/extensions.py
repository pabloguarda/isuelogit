#=============================================================================
# LINK PERFORMANCE FUNCTIONS
#=============================================================================

# Note that there is few work that has presented appropiate methods to fit parameters of the BPR function
# https://onlinelibrary.wiley.com/doi/pdf/10.1002/atr.5670330305
# http://onlinepubs.trb.org/Onlinepubs/trr/1987/1120/1120-004.pdf

# Conical Volume-Delay functions as alternative to BPR? (2001). Interestingly, no reference is given on how to fit those functoins
# https://pubsonline.informs.org/doi/pdf/10.1287/trsc.24.2.153


# i) Read cross sectional pems traffic flow data for the stations and fit the bpr functions
# n = 320 stations in Fresno, although what matters is the hugh amount of longitudinal data



# ii) Perform clustering to divide links according to a set of features available in osm (type of highway ('highway'),'lanes', maximum speed limit ('maxspeed'), 'bridge', 'tunnel'
# n = 37600 segments to perform clustering

#- Kmeans (remember to normalize the means of the attributes)



# iii) Run a linear regression in each cluster



# iv) Assign the bpr parameters depending on the cluster associated to each link

# Note that capacity of the road is rarely available so a conversion factor between lanes and vehicles per lane needs to be estimated. For the street segments where traffic stations are installed, the capacity equates the maximum link flow in the period of analysis. For the remaining street segments, we can impute the average capacity of the station within the clusters,
