from torch import Tensor
import torch


def points_inside_polygon_edges(points: Tensor, edges: Tensor) -> Tensor:
    """A point is inside a polygon if it is inside all the edges' half-planes.
    B sets of N points (B, N, 2), B M-sided polygon edges (B, M, 2, 2) -> (B, N, 2)
    """
    inside_points = points.clone()
    p1, p2 = edges[:, :, 0, :], edges[:, :, 1, :]
    a = (p2[:, :, 1] - p1[:, :, 1]).unsqueeze(1)  # (B, 1, M)
    b = (p1[:, :, 0] - p2[:, :, 0]).unsqueeze(1)
    c = (p2[:, :, 0] * p1[:, :, 1] - p1[:, :, 0] * p2[:, :, 1]).unsqueeze(1)
    points = points.unsqueeze(2)  # (B, N, 1, 2)
    mask = (a * points[..., 0] + b * points[..., 1] + c < 0).all(dim=2)
    inside_points[~mask] = torch.nan
    return inside_points


def segment_intersections(segments1: Tensor, segments2: Tensor) -> Tensor:
    """segments1 and segments2 have shape (N, 2, 2)"""
    A1, A2, B1, B2 = segments1[:, 0], segments1[:, 1], segments2[:, 0], segments2[:, 1]
    V1, V2 = A2 - A1, B2 - B1  # Direction vectors
    det = torch.det(torch.stack([V1.unsqueeze(1), V2.unsqueeze(1)], dim=2))
    parallel_mask = torch.abs(det) < 1e-5  # parallel or collinear
    B1_A1 = B1 - A1
    s = torch.det(torch.stack([B1_A1.unsqueeze(1), V2.unsqueeze(1)], dim=2)) / det
    t = torch.det(torch.stack([B1_A1.unsqueeze(1), V1.unsqueeze(1)], dim=2)) / det
    valid_mask = (~parallel_mask) & ((s >= 0) & (s <= 1)) & ((t >= 0) & (t <= 1))
    intersection_points = A1 + s * V1
    intersection_points[~valid_mask.squeeze()] = torch.nan
    return intersection_points


def polygon_intersection(polygons1: Tensor, polygons2: Tensor) -> Tensor:
    """the pairwise intersection polygons of 2 sets of convex n-gons.
    arguments' shape: (B, n, 2).
    the intersection polygon is made up of intersection points and inside points.
    """
    assert polygons1.shape == polygons2.shape
    num_polygons, num_sides, _ = polygons1.shape
    edges1 = torch.stack([polygons1, torch.roll(polygons1, shifts=1, dims=1)], dim=2)
    edges2 = torch.stack([polygons2, torch.roll(polygons2, shifts=1, dims=1)], dim=2)
    intersections = torch.cat(  # (B, N*N + 2N, 2)
        [
            segment_intersections(
                edges1.repeat_interleave(num_sides, dim=1).reshape(-1, 2, 2),
                edges2.repeat((1, num_sides, 1, 1)).reshape(-1, 2, 2),
            ).reshape(num_polygons, num_sides * num_sides, 2),
            points_inside_polygon_edges(points=polygons1, edges=edges2),
            points_inside_polygon_edges(points=polygons2, edges=edges1),
        ],
        dim=1,
    )
    return intersections


def polygon_area(polygons: Tensor) -> Tensor:
    """Area of polygons (B, n, 2) using shoelace formula"""
    # polygons = reorder_clockwise(polygons)
    valid_mask = ~torch.isnan(polygons).any(dim=-1)
    x, y = polygons[..., 0], polygons[..., 1]
    shift_x = torch.roll(x, shifts=-1, dims=1)
    shift_x[shift_x.isnan()] = shift_x[..., -1:].expand_as(shift_x)[shift_x.isnan()]
    shift_y = torch.roll(y, shifts=-1, dims=1)
    shift_y[shift_y.isnan()] = shift_y[..., -1:].expand_as(shift_y)[shift_y.isnan()]
    x[~valid_mask] = 0
    y[~valid_mask] = 0
    shift_x[~valid_mask] = 0
    shift_y[~valid_mask] = 0
    cross_product = x * shift_y - y * shift_x
    cross_product[~valid_mask] = 0
    area = 0.5 * torch.abs(torch.sum(cross_product, dim=1))
    return area


def polygon_iou(polygons1: Tensor, polygons2: Tensor) -> float:
    """Return intersection-over-union (Jaccard index) between two sets of polygons.

    Both sets of polygons are expected to be in ``[[x, y], ...]`` format

    Args:
        polygons1 (Tensor[N, n, 2]): first set of polygons
        polygons2 (Tensor[M, n, 2]): second set of polygons

    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise IoU values for every element in polygons1 and polygons2
    """
    num_polygons1, num_polygons2 = polygons1.shape[0], polygons2.shape[0]
    polygons1 = polygons1.repeat_interleave(num_polygons2, dim=0)
    polygons2 = polygons2.repeat((num_polygons1, 1, 1))
    intersections = polygon_area(polygon_intersection(polygons1, polygons2))
    unions = polygon_area(polygons1) + polygon_area(polygons2) - intersections
    return (intersections / unions).reshape((num_polygons1, num_polygons2))

