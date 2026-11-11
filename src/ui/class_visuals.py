"""Class-visualization helpers: palette, stacked bar."""
from __future__ import annotations

from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget


CLASS_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
]


def build_class_colors(model) -> dict[str, QColor]:
    """Stable class -> color mapping derived from the root distribution.

    Returns an empty dict for regression trees (no `distribution` key in
    the root summary), which the widgets use as a signal to skip the bar.
    """
    summary = model.root.summary
    if "distribution" not in summary:
        return {}
    classes = sorted(summary["distribution"].keys())
    return {cls: QColor(CLASS_PALETTE[i % len(CLASS_PALETTE)])
            for i, cls in enumerate(classes)}


class StackedBar(QWidget):
    """
    Horizontal stacked bar: one colored segment per class.
    Теперь поддерживает подписи долей в процентах с адаптивным цветом текста.
    """

    def __init__(self, distribution: dict[str, int], colors: dict[str, QColor], parent=None):
        super().__init__(parent)
        self._colors = colors
        total = sum(distribution.values()) or 1
        
        # Сортируем для стабильного порядка (как в легенде)
        self._parts = [(cls, v / total) for cls, v in sorted(distribution.items())]
        
        # Минимальная высота для читаемости текста
        self.setFixedHeight(24)  
        self.setMinimumWidth(140)

    def paintEvent(self, event):
        from PyQt6.QtGui import QFontMetrics
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        
        # Фон бара (опционально, для четких границ)
        bg_color = QColor("#f5f7fa")
        painter.fillRect(0, 0, w, h, bg_color)
        
        x = 0.0
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        fm = QFontMetrics(font)
        
        for cls, frac in self._parts:
            seg_w = w * frac
            color = self._colors.get(cls, QColor("#cccccc"))
            
            # Рисуем сегмент
            painter.fillRect(QRectF(x, 0, seg_w, h), color)
            
            # --- НОВОЕ: Подпись доли ---
            if seg_w > 12:  # Не рисуем текст, если сегмент слишком узкий
                pct = frac * 100
                
                # Форматируем: для малых долей (NPL) показываем 1 знак, для больших — без лишних нулей
                if pct < 10:
                    text = f"{pct:.1f}%"
                else:
                    text = f"{pct:.0f}%"
                
                text_w = fm.horizontalAdvance(text)
                text_h = fm.height()
                
                # Вычисляем центр сегмента
                text_x = x + (seg_w - text_w) / 2
                text_y = (h + text_h) / 2 - 2  # -2 для оптического центрирования
                
                # Выбираем цвет текста: черный на светлом, белый на темном
                # Порог яркости 0.5 в HSV или просто проверка компонентов RGB
                r, g, b = color.red(), color.green(), color.blue()
                brightness = (r * 299 + g * 587 + b * 114) / 1000
                
                if brightness > 140:
                    text_color = QColor("#222222")  # Темный текст на светлом фоне
                else:
                    text_color = QColor("#ffffff")  # Светлый текст на темном фоне
                
                painter.setPen(text_color)
                painter.drawText(int(text_x), int(text_y), text)
            
            x += seg_w
            
        # Рамка всего виджета
        painter.setPen(QPen(QColor("#666666"), 1))
        painter.drawRect(0, 0, w - 1, h - 1)
        painter.end()