using Gridap
using HyperFEM, HyperFEM.ComputationalModels.CartesianTags

function generate_tessellation(; long, width, thick, ndivisions, args...)
  domain = (0.0, long, 0.0, width, 0.0, thick)
  partition = ndivisions .* (8, 3, 2)
  geometry = CartesianDiscreteModel(domain, partition)
  labels = get_face_labeling(geometry)
  add_tag_from_tags!(labels, "bottom", CartesianTags.faceXY0⁺)
  add_tag_from_tags!(labels, "top", CartesianTags.faceXY1⁺)
  add_tag_from_tags!(labels, "fixed", CartesianTags.face0YZ⁺)
  add_tag_from_tags!(labels, "free-end", CartesianTags.face1YZ⁺)
  add_tag_from_vertex_filter!(labels, "mid", geometry, x -> x[3] ≈ 0.5thick)
  geometry
end

problem_data = (
  long = 0.025,  # m
  width = 0.003,
  thick = 0.001,
  ndivisions = 2
)

geometry = generate_tessellation(; problem_data...)

writevtk(geometry, joinpath(dirname(@__FILE__), "geometry"))
