using Gridap
using HyperFEM, HyperFEM.ComputationalModels.CartesianTags

function generate_tessellation(; width, thick0, prestretch, ndivisions, args...)
  λ3 = 1 / prestretch^2
  thick = thick0*λ3
  domain = (-0.5width, 0.5width, -0.5width, 0.5width, 0.0, thick)
  partition = (2*ndivisions, 2*ndivisions, 2)
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

writevtk(geometry, "geometry")
