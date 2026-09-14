"""A red-black tree: a self-balancing binary search tree guaranteeing
O(log n) insert, delete, and search.

The heap in scheduler.py is great for repeatedly pulling the single
earliest job, but it can't efficiently remove or look up an arbitrary
job without a full scan. A red-black tree keeps every key in sorted
order and supports insert, delete, and search of *any* key in O(log n) --
useful if jobs need to be cancelled or rescheduled by id rather than only
ever popped in time order.

The four invariants that keep the tree balanced:
1. Every node is red or black.
2. The root is black.
3. Every leaf (the NIL sentinel) is black.
4. A red node never has a red child.
5. Every path from a node to any descendant NIL leaf passes through the
   same number of black nodes (that node's "black-height").

Together these bound the height at O(log n), which is what keeps every
operation logarithmic even in the worst case -- unlike a plain BST, which
degrades to O(n) on already-sorted input.
"""

from __future__ import annotations

from typing import Any, Iterator

RED = "red"
BLACK = "black"


class _Node:
    __slots__ = ("key", "value", "color", "left", "right", "parent")

    def __init__(self, key, value, color, left=None, right=None, parent=None):
        self.key = key
        self.value = value
        self.color = color
        self.left = left
        self.right = right
        self.parent = parent


class RBTree:
    def __init__(self) -> None:
        # A single shared sentinel stands in for every NIL leaf and for the
        # root's parent, so rotations/fixups never have to special-case
        # "no such node" -- it's always a real (black) node to point at.
        self._nil = _Node(key=None, value=None, color=BLACK)
        self._nil.left = self._nil
        self._nil.right = self._nil
        self._nil.parent = self._nil
        self._root = self._nil
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def __contains__(self, key: Any) -> bool:
        return self._find(key) is not self._nil

    def __iter__(self) -> Iterator[tuple[Any, Any]]:
        yield from self._inorder(self._root)

    def _inorder(self, node: _Node) -> Iterator[tuple[Any, Any]]:
        if node is self._nil:
            return
        yield from self._inorder(node.left)
        yield (node.key, node.value)
        yield from self._inorder(node.right)

    def search(self, key: Any) -> Any | None:
        node = self._find(key)
        return None if node is self._nil else node.value

    def _find(self, key: Any) -> _Node:
        node = self._root
        while node is not self._nil and key != node.key:
            node = node.left if key < node.key else node.right
        return node

    def minimum(self) -> tuple[Any, Any] | None:
        if self._root is self._nil:
            return None
        node = self._min_node(self._root)
        return (node.key, node.value)

    def maximum(self) -> tuple[Any, Any] | None:
        if self._root is self._nil:
            return None
        node = self._root
        while node.right is not self._nil:
            node = node.right
        return (node.key, node.value)

    def _min_node(self, node: _Node) -> _Node:
        while node.left is not self._nil:
            node = node.left
        return node

    def _left_rotate(self, x: _Node) -> None:
        y = x.right
        x.right = y.left
        if y.left is not self._nil:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is self._nil:
            self._root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def _right_rotate(self, x: _Node) -> None:
        y = x.left
        x.left = y.right
        if y.right is not self._nil:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is self._nil:
            self._root = y
        elif x is x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

    def insert(self, key: Any, value: Any = None) -> None:
        existing = self._find(key)
        if existing is not self._nil:
            existing.value = value
            return

        z = _Node(key=key, value=value, color=RED, left=self._nil, right=self._nil, parent=self._nil)
        y = self._nil
        x = self._root
        while x is not self._nil:
            y = x
            x = x.left if z.key < x.key else x.right
        z.parent = y
        if y is self._nil:
            self._root = z
        elif z.key < y.key:
            y.left = z
        else:
            y.right = z

        self._size += 1
        self._insert_fixup(z)

    def _insert_fixup(self, z: _Node) -> None:
        while z.parent.color == RED:
            if z.parent is z.parent.parent.left:
                uncle = z.parent.parent.right
                if uncle.color == RED:
                    z.parent.color = BLACK
                    uncle.color = BLACK
                    z.parent.parent.color = RED
                    z = z.parent.parent
                else:
                    if z is z.parent.right:
                        z = z.parent
                        self._left_rotate(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._right_rotate(z.parent.parent)
            else:
                uncle = z.parent.parent.left
                if uncle.color == RED:
                    z.parent.color = BLACK
                    uncle.color = BLACK
                    z.parent.parent.color = RED
                    z = z.parent.parent
                else:
                    if z is z.parent.left:
                        z = z.parent
                        self._right_rotate(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._left_rotate(z.parent.parent)
        self._root.color = BLACK

    def _transplant(self, u: _Node, v: _Node) -> None:
        if u.parent is self._nil:
            self._root = v
        elif u is u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def delete(self, key: Any) -> bool:
        z = self._find(key)
        if z is self._nil:
            return False

        y = z
        y_original_color = y.color
        if z.left is self._nil:
            x = z.right
            self._transplant(z, z.right)
        elif z.right is self._nil:
            x = z.left
            self._transplant(z, z.left)
        else:
            y = self._min_node(z.right)
            y_original_color = y.color
            x = y.right
            if y.parent is z:
                x.parent = y
            else:
                self._transplant(y, y.right)
                y.right = z.right
                y.right.parent = y
            self._transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color

        if y_original_color == BLACK:
            self._delete_fixup(x)

        self._size -= 1
        return True

    def _delete_fixup(self, x: _Node) -> None:
        while x is not self._root and x.color == BLACK:
            if x is x.parent.left:
                sibling = x.parent.right
                if sibling.color == RED:
                    sibling.color = BLACK
                    x.parent.color = RED
                    self._left_rotate(x.parent)
                    sibling = x.parent.right
                if sibling.left.color == BLACK and sibling.right.color == BLACK:
                    sibling.color = RED
                    x = x.parent
                else:
                    if sibling.right.color == BLACK:
                        sibling.left.color = BLACK
                        sibling.color = RED
                        self._right_rotate(sibling)
                        sibling = x.parent.right
                    sibling.color = x.parent.color
                    x.parent.color = BLACK
                    sibling.right.color = BLACK
                    self._left_rotate(x.parent)
                    x = self._root
            else:
                sibling = x.parent.left
                if sibling.color == RED:
                    sibling.color = BLACK
                    x.parent.color = RED
                    self._right_rotate(x.parent)
                    sibling = x.parent.left
                if sibling.right.color == BLACK and sibling.left.color == BLACK:
                    sibling.color = RED
                    x = x.parent
                else:
                    if sibling.left.color == BLACK:
                        sibling.right.color = BLACK
                        sibling.color = RED
                        self._left_rotate(sibling)
                        sibling = x.parent.left
                    sibling.color = x.parent.color
                    x.parent.color = BLACK
                    sibling.left.color = BLACK
                    self._right_rotate(x.parent)
                    x = self._root
        x.color = BLACK
