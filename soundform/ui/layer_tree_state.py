"""Retain layer-tree navigation while scene rows are rebuilt."""
from contextlib import contextmanager
from PySide6.QtCore import Qt, QItemSelectionModel
from PySide6.QtWidgets import QTreeWidgetItemIterator

STATE_ROLE = Qt.UserRole+3


def _items(tree):
    iterator = QTreeWidgetItemIterator(tree)
    while iterator.value():
        yield iterator.value()
        iterator += 1


def _key(item):
    if item is None:
        return None
    if item.parent() is None:
        return ("layer",item.text(0))
    return item.data(0,STATE_ROLE)


def capture_tree(tree):
    return dict(expanded={_key(item) for item in _items(tree) if item.isExpanded()},
                selected={_key(item) for item in tree.selectedItems()},
                current=_key(tree.currentItem()),
                vertical=tree.verticalScrollBar().value(),
                horizontal=tree.horizontalScrollBar().value())


def restore_tree(tree,state):
    if state is None:
        return
    current = None
    for item in _items(tree):
        key = _key(item)
        item.setExpanded(key in state["expanded"])
        item.setSelected(key in state["selected"])
        if key == state["current"]:
            current = item
    if current is not None:
        tree.setCurrentItem(current,0,QItemSelectionModel.NoUpdate)
    tree.doItemsLayout()
    tree.verticalScrollBar().setValue(state["vertical"])
    tree.horizontalScrollBar().setValue(state["horizontal"])


@contextmanager
def preserve_tree(tree,reset=False):
    state = None if reset else capture_tree(tree)
    blocked = tree.blockSignals(True)
    updates = tree.updatesEnabled()
    tree.setUpdatesEnabled(False)
    try:
        yield
    finally:
        restore_tree(tree,state)
        tree.setUpdatesEnabled(updates)
        tree.blockSignals(blocked)
