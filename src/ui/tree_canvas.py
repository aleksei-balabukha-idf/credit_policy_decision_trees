"""Tree canvas: QGraphicsView with zoom / pan / PNG export and a simple
tidy level-based layout."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen
from PyQt6.QtWidgets import (QGraphicsLineItem, QGraphicsRectItem,
                             QGraphicsScene, QGraphicsTextItem,
                             QGraphicsView)

from core.tree import DecisionTreeModel, Node
from .node_widget import NodeWidget
from .class_visuals import build_class_colors

NODE_W, NODE_H = NodeWidget.WIDTH + 20, 210       # стартовые оценки
HGAP, VGAP = 40, 90


class TreeCanvas(QGraphicsView):
    split_best_clicked = pyqtSignal(int)
    split_clicked = pyqtSignal(int)
    undo_split_clicked = pyqtSignal(int)
    nan_route_changed = pyqtSignal(int, str)          

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)   # pan by drag
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.model: DecisionTreeModel | None = None

    # --------------------------------------------------------- navigation

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def zoom_fit(self):
        if self._scene.items():
            self.fitInView(self._scene.itemsBoundingRect(),
                           Qt.AspectRatioMode.KeepAspectRatio)

    # ------------------------------------------------------------- render

    def rebuild(self, model: DecisionTreeModel):
        self.model = model
        self._scene.clear()

        # 1) цвета классов из распределения в корне
        colors = build_class_colors(model)

        # 2) первичная раскладка по слотам / глубине
        pos: dict[int, tuple[float, float]] = {}
        self._layout(model.root, [0], pos)

        # 3) создаём карточки, запоминаем реальные высоты
        proxies: dict[int, object] = {}
        heights: dict[int, int] = {}
        for nid, (x, y) in pos.items():
            node = model.get_node(nid)
            w = NodeWidget(node, model.features, colors)
            w.split_best_clicked.connect(self.split_best_clicked)
            w.split_clicked.connect(self.split_clicked)
            w.undo_split_clicked.connect(self.undo_split_clicked)
            w.nan_route_changed.connect(self.nan_route_changed)
            proxy = self._scene.addWidget(w)
            proxy.setPos(x, y)
            proxies[nid] = proxy
            heights[nid] = w.sizeHint().height()

        # 4) пересчёт Y — карточки разной высоты не должны пересекаться
        depth_h: dict[int, int] = {}
        for nid in pos:
            d = model.get_node(nid).depth
            depth_h[d] = max(depth_h.get(d, 0), heights[nid])
        depth_y = {0: 0.0}
        prev = 0
        for d in sorted(depth_h):
            if d != 0:
                depth_y[d] = depth_y[prev] + depth_h[prev] + VGAP
            prev = d
        for nid, (x, _y) in list(pos.items()):
            d = model.get_node(nid).depth
            new_y = depth_y.get(d, 0.0)
            pos[nid] = (x, new_y)
            proxies[nid].setPos(x, new_y)

        # 5) рёбра + подписи
        self._draw_edges(model.root, pos, heights)

        # 6) легенда над корнем — прямыми items, без proxy-виджета
        if colors:
            rx, _ry = pos[model.root.id]
            self._add_legend(colors, rx, -70.0)

        self._scene.setSceneRect(self._scene.itemsBoundingRect().adjusted(
            -50, -50, 50, 50))

    # --------------------------------------------------------------------

    def _layout(self, node: Node, counter: list[int],
                pos: dict[int, tuple[float, float]]) -> tuple[float, float]:
        """In-order slot assignment; returns (x_start_slot, x_end_slot)."""
        if node.is_leaf:
            slot = counter[0]
            counter[0] += 1
            pos[node.id] = (slot * (NODE_W + HGAP),
                            node.depth * (NODE_H + VGAP))
            return slot, slot
        start, end = None, None
        for child in node.children.values():
            s, e = self._layout(child, counter, pos)
            start = s if start is None else min(start, s)
            end = e if end is None else max(end, e)
        x = ((start + end) / 2) * (NODE_W + HGAP)
        pos[node.id] = (x, node.depth * (NODE_H + VGAP))
        return start, end

    def _draw_edges(self, node: Node, pos, heights: dict[int, int]):
        x1, y1 = pos[node.id]
        h1 = heights.get(node.id, NODE_H)
        for side, child in node.children.items():
            x2, y2 = pos[child.id]
            line = QGraphicsLineItem(x1 + NODE_W / 2, y1 + h1,
                                     x2 + NODE_W / 2, y2)
            line.setPen(QPen(QColor("#7f8fa6"), 2))
            self._scene.addItem(line)
            label = QGraphicsTextItem(child.rule)
            label.setDefaultTextColor(QColor("#555f6e"))
            label.setPos((x1 + x2) / 2 + NODE_W / 4 - 40,
                         (y1 + h1 + y2) / 2 - 24)
            self._scene.addItem(label)
            self._draw_edges(child, pos, heights)

    def _add_legend(self, colors: dict, x: float, y: float) -> None:
        """Draw the class legend as direct scene items.

        Background is sized from the ACTUAL laid-out content: we place
        title/labels/swatches first, measure the rightmost edge, and only
        then draw the background behind them (setZValue(-1)). This avoids
        the previous bug where the width formula and the layout disagreed
        on the number of `gap`s and clipped the last class label.
        """
        from PyQt6.QtGui import QBrush, QFont

        pad, gap, swatch = 10.0, 8.0, 14.0
        text_color = QColor("#222222")
        border_pen = QPen(QColor("#b6c2d2"), 1)

        # ---- title ------------------------------------------------
        title = QGraphicsTextItem("Classes:")
        f = QFont(title.font())
        f.setBold(True)
        title.setFont(f)
        title.setDefaultTextColor(text_color)
        title.setPos(x + pad, y + pad)
        self._scene.addItem(title)

        title_w = title.boundingRect().width()
        title_h = title.boundingRect().height()
        row_h = max(swatch, title_h)

        # ---- one swatch + label per class -------------------------
        cur_x = x + pad + title_w + gap * 2
        for cls in colors:
            # swatch, vertically centered in the row
            rect = QGraphicsRectItem(
                cur_x, y + pad + (row_h - swatch) / 2, swatch, swatch)
            rect.setBrush(QBrush(colors[cls]))
            rect.setPen(QPen(QColor("#666666"), 1))
            self._scene.addItem(rect)
            cur_x += swatch + gap

            # label, vertically centered in the row
            lbl = QGraphicsTextItem(str(cls))
            lbl.setDefaultTextColor(text_color)
            lbl_h = lbl.boundingRect().height()
            lbl.setPos(cur_x, y + pad + (row_h - lbl_h) / 2)
            self._scene.addItem(lbl)
            cur_x += lbl.boundingRect().width() + gap * 2

        # ---- background from real content bounds ------------------
        # cur_x currently points past the trailing gap*2 of the last
        # label; drop that trailing gap, then add `pad` on the right.
        content_right = cur_x - gap * 2
        total_w = (content_right - x) + pad
        total_h = row_h + pad * 2

        bg = QGraphicsRectItem(x, y, total_w, total_h)
        bg.setBrush(QBrush(QColor("#ffffff")))
        bg.setPen(border_pen)
        bg.setZValue(-1)                      # behind title / labels / swatches
        self._scene.addItem(bg)

    # ------------------------------------------------------------ export

    def export_png(self, path: str) -> None:
        rect = self._scene.itemsBoundingRect().adjusted(-20, -20, 20, 20)
        img = QImage(int(rect.width()), int(rect.height()),
                     QImage.Format.Format_ARGB32)
        img.fill(Qt.GlobalColor.white)
        painter = QPainter(img)
        self._scene.render(painter, target=rect)
        painter.end()
        img.save(path, "PNG")