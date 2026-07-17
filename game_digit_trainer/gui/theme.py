"""简洁工具风样式（浅色、大按钮、少干扰）。"""

APP_QSS = """
QMainWindow, QWidget {
  background: #f4f5f7;
  color: #1f2937;
  font-size: 13px;
}
QStatusBar {
  background: #eef0f3;
  color: #4b5563;
}
QTabWidget::pane {
  border: 1px solid #d7dbe2;
  border-radius: 8px;
  background: #ffffff;
  top: -1px;
}
QTabBar::tab {
  background: #e8eaef;
  color: #4b5563;
  padding: 10px 18px;
  margin-right: 4px;
  border-top-left-radius: 8px;
  border-top-right-radius: 8px;
  min-width: 88px;
}
QTabBar::tab:selected {
  background: #ffffff;
  color: #111827;
  font-weight: 600;
}
QTabBar::tab:!selected:hover {
  background: #dde1e8;
}
QPushButton {
  background: #ffffff;
  border: 1px solid #c9ced6;
  border-radius: 8px;
  padding: 8px 14px;
  min-height: 20px;
}
QPushButton:hover {
  background: #f8fafc;
  border-color: #9aa3b2;
}
QPushButton:pressed {
  background: #eef2f7;
}
QPushButton#primaryBtn {
  background: #2563eb;
  color: white;
  border: none;
  font-weight: 600;
  padding: 10px 18px;
}
QPushButton#primaryBtn:hover {
  background: #1d4ed8;
}
QPushButton#successBtn {
  background: #059669;
  color: white;
  border: none;
  font-weight: 700;
  font-size: 16px;
  padding: 14px 20px;
}
QPushButton#successBtn:hover {
  background: #047857;
}
QPushButton#successBtn:disabled {
  background: #9ca3af;
}
QPushButton#dangerBtn {
  background: #fff;
  color: #b91c1c;
  border: 1px solid #f1a8a8;
}
QPushButton#digitBtn {
  font-size: 20px;
  font-weight: 700;
  min-width: 52px;
  min-height: 52px;
  padding: 0;
}
QPushButton#unitBtn {
  font-size: 16px;
  font-weight: 600;
  min-height: 44px;
}
QLineEdit, QSpinBox, QComboBox, QTextEdit, QListWidget {
  background: #ffffff;
  border: 1px solid #c9ced6;
  border-radius: 8px;
  padding: 6px 8px;
  selection-background-color: #2563eb;
}
QListWidget::item {
  padding: 6px 8px;
}
QListWidget::item:selected {
  background: #dbeafe;
  color: #1e3a8a;
}
QGroupBox {
  background: #ffffff;
  border: 1px solid #d7dbe2;
  border-radius: 10px;
  margin-top: 12px;
  padding: 12px 10px 10px 10px;
  font-weight: 600;
}
QGroupBox::title {
  subcontrol-origin: margin;
  left: 12px;
  padding: 0 6px;
  color: #374151;
}
QLabel#hintLabel {
  color: #6b7280;
  font-size: 12px;
}
QLabel#titleLabel {
  font-size: 15px;
  font-weight: 700;
  color: #111827;
}
QLabel#stepLabel {
  color: #2563eb;
  font-weight: 600;
}
QFrame#topBar {
  background: #ffffff;
  border: 1px solid #d7dbe2;
  border-radius: 10px;
}
QLabel#badge {
  background: #fee2e2;
  color: #991b1b;
  border-radius: 10px;
  padding: 2px 8px;
  font-weight: 700;
}
QPushButton#badgeBtn {
  background: #fee2e2;
  color: #991b1b;
  border: 1px solid #fecaca;
  border-radius: 10px;
  padding: 6px 12px;
  font-weight: 700;
}
QPushButton#badgeBtn:hover {
  background: #fecaca;
}
"""
