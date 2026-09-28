"""
Grid Dijkstra for the forward Finsler distance d_F(x, y) (minimum-time-to-reach) on a rectangle.

Directed graph on a regular grid with 32 neighbour directions (all offsets with max-norm <= 3 and
coprime components); edge cost = F(midpoint, x_j - x_i), which is asymmetric.  Dijkstra then gives
the min-time value function from any set of sources to all nodes.  Discretisation overestimates the
true distance slightly (angle quantisation <= ~10 degrees, midpoint quadrature); with h = 0.04 and
fields varying on the scale 0.8 the error is at the percent level (checked against BVP solutions in
experiments/fm_planner.py).  Robust where shooting BVPs fail (multiple branches, long paths).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

__all__ = ["GridFinslerDistance", "GridSE2Distance"]


class GridFinslerDistance:
    def __init__(self, F, xlim, ylim, h=0.04, radius=3):
        self.F = F
        self.xs = np.arange(xlim[0], xlim[1] + 1e-9, h)
        self.ys = np.arange(ylim[0], ylim[1] + 1e-9, h)
        self.h = h
        nx, ny = len(self.xs), len(self.ys)
        self.shape = (nx, ny)
        X, Y = np.meshgrid(self.xs, self.ys, indexing="ij")
        self.nodes = np.stack([X.ravel(), Y.ravel()], -1)                 # node id = ix * ny + iy
        offs = [(i, j) for i in range(-radius, radius + 1) for j in range(-radius, radius + 1)
                if (i, j) != (0, 0) and np.gcd(abs(i), abs(j)) == 1]
        rows, cols, costs = [], [], []
        ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        ix, iy = ix.ravel(), iy.ravel()
        for (i, j) in offs:
            jx, jy = ix + i, iy + j
            ok = (jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny)
            src = ix[ok] * ny + iy[ok]
            dst = jx[ok] * ny + jy[ok]
            d = np.array([i * h, j * h])
            mid = 0.5 * (self.nodes[src] + self.nodes[dst])
            c = F.F(mid, np.broadcast_to(d, mid.shape))
            rows.append(src)
            cols.append(dst)
            costs.append(c)
        rows, cols, costs = np.concatenate(rows), np.concatenate(cols), np.concatenate(costs)
        N = nx * ny
        self.graph = coo_matrix((costs, (rows, cols)), shape=(N, N)).tocsr()
        self.n_edges = len(costs)

    def node_of(self, pts):
        pts = np.atleast_2d(np.asarray(pts, float))
        ix = np.clip(np.rint((pts[:, 0] - self.xs[0]) / self.h).astype(int), 0, self.shape[0] - 1)
        iy = np.clip(np.rint((pts[:, 1] - self.ys[0]) / self.h).astype(int), 0, self.shape[1] - 1)
        return ix * self.shape[1] + iy

    def dist_from(self, sources):
        """Forward distances d_F(source_k, .) for all nodes: array (n_sources, N)."""
        return dijkstra(self.graph, directed=True, indices=self.node_of(sources))

    def dist(self, starts, ends):
        """d_F(start_k, end_k) for paired arrays (n, 2), (n, 2)."""
        D = self.dist_from(starts)
        return D[np.arange(len(D)), self.node_of(ends)]

    def dist_to_target(self, target):
        """d_F(., target) for all nodes: one Dijkstra on the transposed (reversed) graph."""
        return dijkstra(self.graph.T.tocsr(), directed=True, indices=self.node_of(target))[0]

    def dist_to_targets(self, starts, targets):
        """d_F(start_k, target_m): array (n_starts, n_targets)."""
        return self.dist_from(starts)[:, self.node_of(targets)]


class GridSE2Distance:
    """Grid Dijkstra for a left-invariant Finsler metric on SE(2): nodes (x, y, theta) with periodic theta.

    Neighbour offsets: all (i, j, k) with max-norm <= radius and gcd = 1 (98 directions for radius 2).
    Edge cost = F(midpoint, dq) with dq = (i h, j h, k h_theta).  Coarser than the planar grid
    (h ~ 0.1, h_theta ~ 10 degrees), so treat distances as a reference with a few percent tolerance.
    """

    def __init__(self, F, xlim, ylim, h=0.1, n_theta=36, radius=2):
        self.F = F
        self.xs = np.arange(xlim[0], xlim[1] + 1e-9, h)
        self.ys = np.arange(ylim[0], ylim[1] + 1e-9, h)
        self.ths = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
        self.h, self.hth = h, 2 * np.pi / n_theta
        nx, ny, nt = len(self.xs), len(self.ys), n_theta
        self.shape = (nx, ny, nt)
        X, Y, TH = np.meshgrid(self.xs, self.ys, self.ths, indexing="ij")
        self.nodes = np.stack([X.ravel(), Y.ravel(), TH.ravel()], -1)
        offs = [(i, j, k) for i in range(-radius, radius + 1) for j in range(-radius, radius + 1) for k in range(-radius, radius + 1)
                if (i, j, k) != (0, 0, 0) and np.gcd.reduce([abs(i), abs(j), abs(k)]) == 1]
        ix, iy, it = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nt), indexing="ij")
        ix, iy, it = ix.ravel(), iy.ravel(), it.ravel()
        rows, cols, costs = [], [], []
        for (i, j, k) in offs:
            jx, jy, jt = ix + i, iy + j, (it + k) % nt
            ok = (jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny)
            src = (ix[ok] * ny + iy[ok]) * nt + it[ok]
            dst = (jx[ok] * ny + jy[ok]) * nt + jt[ok]
            dq = np.array([i * h, j * h, k * self.hth])
            mid = self.nodes[src].copy()
            mid[:, :2] += 0.5 * dq[:2]
            mid[:, 2] += 0.5 * dq[2]
            c = F.F(mid, np.broadcast_to(dq, mid.shape))
            rows.append(src)
            cols.append(dst)
            costs.append(c)
        rows, cols, costs = np.concatenate(rows), np.concatenate(cols), np.concatenate(costs)
        N = nx * ny * nt
        self.graph = coo_matrix((costs, (rows, cols)), shape=(N, N)).tocsr()
        self.n_edges = len(costs)

    def node_of(self, pts):
        pts = np.atleast_2d(np.asarray(pts, float))
        nx, ny, nt = self.shape
        ix = np.clip(np.rint((pts[:, 0] - self.xs[0]) / self.h).astype(int), 0, nx - 1)
        iy = np.clip(np.rint((pts[:, 1] - self.ys[0]) / self.h).astype(int), 0, ny - 1)
        it = np.rint((np.remainder(pts[:, 2] + np.pi, 2 * np.pi) - np.pi - self.ths[0]) / self.hth).astype(int) % nt
        return (ix * ny + iy) * nt + it

    def dist_from(self, sources):
        return dijkstra(self.graph, directed=True, indices=self.node_of(sources))

    def dist(self, starts, ends):
        D = self.dist_from(starts)
        return D[np.arange(len(D)), self.node_of(ends)]

    def dist_to_target(self, target):
        """d_F(., target) for all nodes: one Dijkstra on the transposed (reversed) graph."""
        return dijkstra(self.graph.T.tocsr(), directed=True, indices=self.node_of(target))[0]
