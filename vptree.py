"""
This module contains an implementation of a Vantage Point-tree (VP-tree).
Original implementation from https://github.com/RickardSjogren/vptree/blob/master/vptree.py
"""
import math
import statistics as stats

class VPTree:

    """ VP-Tree data structure for efficient nearest neighbor search.

    The VP-tree is a data structure for efficient nearest neighbor
    searching and finds the nearest neighbor in O(log n)
    complexity given a tree constructed of n data points. Construction
    complexity is O(n log n).

    Parameters
    ----------
    points : Iterable
        Construction points.
    dist_fn : Callable
        Function taking to point instances as arguments and returning
        the distance between them.
    root: None
        Do not use. Internal to node creation.
    """
    __slots__ = ["left", "right", "left_min", "left_max", "right_min", "right_max", "dist_fn", "vp"]
    def __init__(self, points, dist_fn, root=None):
        self.left = None
        self.right = None
        self.left_min = math.inf
        self.left_max = 0
        self.right_min = math.inf
        self.right_max = 0
        self.dist_fn = dist_fn

        if not len(points):
            raise ValueError('Points can not be empty.')

        # Vantage point is point furthest from parent vp.
        vp_i = 0
        self.vp = points[0]
        points = points[1:]

        if len(points) == 0:
            return

        # Choose division boundary at median of distances.
        distances = [self.dist_fn(self.vp, p) for p in points]
        median = stats.median(distances)

        left_points = []
        right_points = []
        for point, distance in zip(points, distances):
            if distance >= median:
                self.right_min = min(distance, self.right_min)
                if distance > self.right_max:
                    self.right_max = distance
                    right_points.insert(0, point) # put furthest first
                else:
                    right_points.append(point)
            else:
                self.left_min = min(distance, self.left_min)
                if distance > self.left_max:
                    self.left_max = distance
                    left_points.insert(0, point) # put furthest first
                else:
                    left_points.append(point)

        # Modified from the original implementation, removing recursive node creation as large trees cannot be constructed otherwise.
        if root is not None:
            if len(left_points) > 0:
                root.append((self, 0, left_points))

            if len(right_points) > 0:
                root.append((self, 1, right_points))
        elif root is None:
            accumulator = []

            if len(left_points) > 0:
                self.left = VPTree(points=left_points, dist_fn=self.dist_fn, root=accumulator)

            if len(right_points) > 0:
                self.right = VPTree(points=right_points, dist_fn=self.dist_fn, root=accumulator)

            while len(accumulator) > 0:
                node, direction, points = accumulator.pop()
                if direction == 0:
                    node.left = VPTree(points=points, dist_fn=self.dist_fn, root=accumulator)
                elif direction == 1:
                    node.right = VPTree(points=points, dist_fn=self.dist_fn, root=accumulator)
        else:
            raise TypeError("Invalid type applied to the root node. Pass \'None\' for the root node")


    def _is_leaf(self):
        return (self.left is None) and (self.right is None)

    def get_nearest_neighbor(self, query):
        """ Get single nearest neighbor.

        Parameters
        ----------
        query : Any
            Query point.

        Returns
        -------
        Any
            Single nearest neighbor.
        """
        return self.get_n_nearest_neighbors(query, n_neighbors=1)[0]

    def get_n_nearest_neighbors(self, query, n_neighbors):
        """ Get `n_neighbors` nearest neigbors to `query`

        Parameters
        ----------
        query : Any
            Query point.
        n_neighbors : int
            Number of neighbors to fetch.

        Returns
        -------
        list
            List of `n_neighbors` nearest neighbors.
        """
        if not isinstance(n_neighbors, int) or n_neighbors < 1:
            raise ValueError('n_neighbors must be strictly positive integer')
        neighbors = _AutoSortingList(max_size=n_neighbors)
        nodes_to_visit = [(self, 0)]

        furthest_d = math.inf

        while len(nodes_to_visit) > 0:
            node, d0 = nodes_to_visit.pop(0)
            if node is None or d0 > furthest_d:
                continue

            d = self.dist_fn(query, node.vp)
            if d < furthest_d:
                neighbors.append((d, node.vp))
                furthest_d, _ = neighbors[-1]

            if node._is_leaf():
                continue

            if node.left_min <= d <= node.left_max:
                nodes_to_visit.insert(0, (node.left, 0))
            elif node.left_min - furthest_d <= d <= node.left_max + furthest_d:
                nodes_to_visit.append((node.left,
                                       node.left_min - d if d < node.left_min
                                       else d - node.left_max))

            if node.right_min <= d <= node.right_max:
                nodes_to_visit.insert(0, (node.right, 0))
            elif node.right_min - furthest_d <= d <= node.right_max + furthest_d:
                nodes_to_visit.append((node.right,
                                       node.right_min - d if d < node.right_min
                                       else d - node.right_max))

        return list(neighbors)

    def get_all_in_range(self, query, max_distance):
        """ Find all neighbours within `max_distance`.

        Parameters
        ----------
        query : Any
            Query point.
        max_distance : float
            Threshold distance for query.

        Returns
        -------
        neighbors : list
            List of points within `max_distance`.
        """
        neighbors = list()
        nodes_to_visit = [(self, 0)]

        while len(nodes_to_visit) > 0:
            node, d0 = nodes_to_visit.pop(0)
            if node is None or d0 > max_distance:
                continue

            d = self.dist_fn(query, node.vp)
            if d < max_distance:
                neighbors.append((d, node.vp))

            if node._is_leaf():
                continue

            if node.left_min <= d <= node.left_max:
                nodes_to_visit.insert(0, (node.left, 0))
            elif node.left_min - max_distance <= d <= node.left_max + max_distance:
                nodes_to_visit.append((node.left,
                                       node.left_min - d if d < node.left_min
                                       else d - node.left_max))

            if node.right_min <= d <= node.right_max:
                nodes_to_visit.insert(0, (node.right, 0))
            elif node.right_min - max_distance <= d <= node.right_max + max_distance:
                nodes_to_visit.append((node.right,
                                       node.right_min - d if d < node.right_min
                                       else d - node.right_max))

        return neighbors


class _AutoSortingList(list):

    """ Simple auto-sorting list.

    Inefficient for large sizes since the queue is sorted at
    each push.

    Parameters
    ---------
    size : int, optional
        Max queue size.
    """

    __slots__ = ["max_size"]
    def __init__(self, max_size=None, *args):
        super(_AutoSortingList, self).__init__(*args)
        self.max_size = max_size

    def append(self, item):
        """ Append `item` and sort.

        Parameters
        ----------
        item : Any
            Input item.
        """
        super(_AutoSortingList, self).append(item)
        self.sort(key=lambda item: item[0])
        if self.max_size is not None and len(self) > self.max_size:
            self.pop()

"""
Copyright 2017 Rickard Sjögren
Copyright 2019 Luke Barr

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""