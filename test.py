import cugraph
import cudf

gdf = cudf.read_csv('./datasets/netscience.csv', delimiter=' ',
                  dtype=['int32', 'int32', 'float32'], header=None)
G = cugraph.Graph()
G.from_cudf_edgelist(gdf, source='0', destination='1', edge_attr= '2')
df, offsets = cugraph.random_walks(G, [3, 9, 5], 3)
print(df)
print("\n\n")
print(offsets, "\n\n")
print(gdf)
