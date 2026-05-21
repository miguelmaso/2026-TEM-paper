using Gridap

module CartesianTags
  const faceX0 = [1, 3, 5, 7, 13, 15, 17, 19, 25]
  const faceX1 = [2, 4, 6, 8, 14, 16, 18, 20, 26]
  const faceY0 = [1, 2, 5, 6, 9, 11, 17, 18, 23]
  const faceY1 = [3, 4, 7, 8, 10, 12, 19, 20, 24]
  const faceZ0 = [1, 2, 3, 4, 9, 10, 13, 14, 21]
  const faceZ1 = [5, 6, 7, 8, 11, 12, 15, 16, 22]
  const edgeX00 = [9]
  const edgeX10 = [10]
  const edgeX01 = [11]
  const edgeX11 = [12]
  const edge0Y0 = [13]
  const edge1Y0 = [14]
  const edge0Y1 = [15]
  const edge1Y1 = [16]
  const edge00Z = [17]
  const edge10Z = [18]
  const edge01Z = [19]
  const edge11Z = [20]
  const corner000 = [1]
  const corner100 = [2]
  const corner010 = [3]
  const corner110 = [4]
  const corner001 = [5]
  const corner101 = [6]
  const corner011 = [7]
  const corner111 = [8]
end

function add_tag_from_vertex_filter!(labels::Gridap.Geometry.FaceLabeling, geometry::Gridap.Geometry.DiscreteModel, tag::String, filter::Function)
  new_labels = Gridap.Geometry.face_labeling_from_vertex_filter(geometry.grid_topology, tag, filter)
  merge!(labels, new_labels)
end

function generate_tessellation(; width, thick0, prestretch, ndivisions, args...)
  λ3 = 1 / prestretch^2
  thick = thick0*λ3
  domain = (-0.5width, 0.5width, -0.5width, 0.5width, 0.0, thick)
  partition = (2*ndivisions, 2*ndivisions, ndivisions/5)
  geometry = CartesianDiscreteModel(domain, partition)
  labels = get_face_labeling(geometry)
  add_tag_from_tags!(labels, "top", CartesianTags.faceZ1)
  add_tag_from_tags!(labels, "bottom", CartesianTags.faceZ0)
  add_tag_from_tags!(labels, "edge", CartesianTags.edgeX00)
  add_tag_from_tags!(labels, "corner", CartesianTags.corner000)
  add_tag_from_tags!(labels, "faces", [CartesianTags.faceX0; CartesianTags.faceX1; CartesianTags.faceY0; CartesianTags.faceY1])
  add_tag_from_vertex_filter!(labels, geometry, "top_electrode",    p -> p[3] ≈ thick && abs(p[1]) <= 0.25width+1e-6 && abs(p[2]) <= 0.25width+1e-6)
  add_tag_from_vertex_filter!(labels, geometry, "bottom_electrode", p -> p[3] ≈ 0.0   && abs(p[1]) <= 0.25width+1e-6 && abs(p[2]) <= 0.25width+1e-6)
  geometry
end

problem_data = (
  width = 0.05,     # 5cm (frame dimensions)
  thick0 = 0.0005,  # 0.5mm (undeformed)
  voltage = 3500,   # V
  prestretch = 1.5, # -
  ndivisions = 10,  # -
  order = 2         # -
)
geometry = generate_tessellation(; problem_data...)

writevtk(geometry, "geomery")
