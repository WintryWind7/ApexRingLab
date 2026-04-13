"""毒圈可视化工具"""

import sys
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QScrollArea
)
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QImage
from PIL import Image, ImageDraw


# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 地图配置（与 Predictor 的 MAP_TO_ONEHOT 保持一致）
MAP_CONFIG = {
    "mp_rr_district": {
        "name": "District",
        "path": PROJECT_ROOT / "ApexRingLab/data/map/mp_rr_district.png",
        "rings": [4930, 2419, 1488]
    },
    "mp_rr_tropic": {
        "name": "Tropic Island",
        "path": PROJECT_ROOT / "ApexRingLab/data/map/mp_rr_tropic_island_mu2.png",
        "rings": [4894, 2407, 1284]
    }
}

# 坐标系统配置
COORD_SIZE = 16384  # 坐标系统尺寸
IMAGE_SIZE = 4096   # 图片尺寸
COORD_TO_IMAGE_SCALE = IMAGE_SIZE / COORD_SIZE  # 0.25


class MapLabel(QLabel):
    """可点击和可缩放的地图标签"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(False)
    
    def mousePressEvent(self, event):
        """鼠标点击事件"""
        if event.button() == Qt.LeftButton and self.pixmap():
            click_pos = event.pos()
            pixmap = self.pixmap()
            label_width = self.width()
            label_height = self.height()
            pixmap_width = pixmap.width()
            pixmap_height = pixmap.height()
            
            # 图片居中显示，计算偏移量
            x_offset = (label_width - pixmap_width) // 2
            y_offset = (label_height - pixmap_height) // 2
            
            # 转换为显示图片坐标
            display_x = click_pos.x() - x_offset
            display_y = click_pos.y() - y_offset
            
            # 检查是否在图片范围内
            if 0 <= display_x < pixmap_width and 0 <= display_y < pixmap_height:
                # 通知父窗口（传递显示坐标）
                if self.parent_window:
                    self.parent_window.on_map_clicked(display_x, display_y)
    
    def wheelEvent(self, event):
        """鼠标滚轮缩放事件"""
        if self.parent_window:
            delta = event.angleDelta().y()
            if delta > 0:
                self.parent_window.zoom_in()
            else:
                self.parent_window.zoom_out()


class RingViewerWindow(QMainWindow):
    """毒圈可视化主窗口"""
    
    def __init__(self):
        super().__init__()
        self.current_map = None
        self.current_ring_level = 1  # 默认第1级毒圈
        self.zoom_scale = 0.5  # 初始缩放比例（50%显示）
        self.original_image = None  # 原始地图图片
        self.ring_data = None  # 当前毒圈数据 (coord_x, coord_y, radius)
        self.init_ui()
    
    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("Apex 毒圈可视化工具")
        self.setGeometry(100, 100, 1000, 900)
        
        # 主窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 控制面板
        control_layout = QHBoxLayout()
        
        # 地图选择
        map_label = QLabel("选择地图:")
        self.map_combo = QComboBox()
        for map_id, config in MAP_CONFIG.items():
            self.map_combo.addItem(config["name"], map_id)
        self.map_combo.currentIndexChanged.connect(self.on_map_changed)
        
        # 毒圈等级选择
        ring_label = QLabel("毒圈等级:")
        self.ring_combo = QComboBox()
        self.ring_combo.addItems(["Ring 1", "Ring 2", "Ring 3"])
        self.ring_combo.currentIndexChanged.connect(self.on_ring_level_changed)
        
        # 清除按钮
        clear_btn = QPushButton("清除毒圈")
        clear_btn.clicked.connect(self.clear_ring)
        
        # 缩放按钮
        zoom_in_btn = QPushButton("放大 (+)")
        zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_out_btn = QPushButton("缩小 (-)")
        zoom_out_btn.clicked.connect(self.zoom_out)
        
        control_layout.addWidget(map_label)
        control_layout.addWidget(self.map_combo)
        control_layout.addWidget(ring_label)
        control_layout.addWidget(self.ring_combo)
        control_layout.addWidget(clear_btn)
        control_layout.addWidget(zoom_in_btn)
        control_layout.addWidget(zoom_out_btn)
        control_layout.addStretch()
        
        # 坐标显示
        self.coord_label = QLabel("坐标: 点击地图以获取坐标 | 缩放: 50% | 滚轮可缩放")
        self.coord_label.setStyleSheet("font-size: 14px; padding: 5px;")
        
        # 地图显示（使用滚动区域）
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setAlignment(Qt.AlignCenter)
        self.map_label = MapLabel(self)
        scroll_area.setWidget(self.map_label)
        
        # 添加到主布局
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.coord_label)
        main_layout.addWidget(scroll_area)
        
        # 加载默认地图
        self.load_map(self.map_combo.currentData())
    
    def load_map(self, map_id: str):
        """加载地图"""
        if map_id not in MAP_CONFIG:
            return
        
        self.current_map = map_id
        map_path = MAP_CONFIG[map_id]["path"]
        
        if not map_path.exists():
            self.coord_label.setText(f"错误: 地图文件不存在 - {map_path}")
            return
        
        # 加载原始图片
        self.original_image = Image.open(str(map_path)).convert("RGBA")
        self.ring_data = None
        
        # 显示缩放后的地图
        self.update_display()
    
    def on_map_changed(self, index):
        """地图切换事件"""
        map_id = self.map_combo.currentData()
        self.load_map(map_id)
    
    def on_ring_level_changed(self, index):
        """毒圈等级切换事件"""
        self.current_ring_level = index + 1
    
    def on_map_clicked(self, display_x: int, display_y: int):
        """地图点击事件（接收显示坐标）"""
        if not self.current_map or not self.original_image:
            return
        
        # 转换显示坐标到原始图片坐标
        original_x = display_x / self.zoom_scale
        original_y = display_y / self.zoom_scale
        
        # 转换为坐标系统坐标
        coord_x = int(original_x / COORD_TO_IMAGE_SCALE)
        coord_y = int(original_y / COORD_TO_IMAGE_SCALE)
        
        # 获取当前毒圈半径
        ring_radius = MAP_CONFIG[self.current_map]["rings"][self.current_ring_level - 1]
        
        # 保存毒圈数据
        self.ring_data = (coord_x, coord_y, ring_radius)
        
        # 更新显示
        self.update_display()
        
        # 更新坐标显示
        self.coord_label.setText(
            f"坐标: ({coord_x}, {coord_y}) | "
            f"毒圈等级: {self.current_ring_level} | "
            f"半径: {ring_radius} | "
            f"缩放: {int(self.zoom_scale * 100)}%"
        )
    
    def update_display(self):
        """更新地图显示（包含缩放和毒圈）"""
        if not self.original_image:
            return
        
        # 复制原始图片
        img = self.original_image.copy()
        
        # 如果有毒圈数据，绘制毒圈
        if self.ring_data:
            coord_x, coord_y, radius = self.ring_data
            draw = ImageDraw.Draw(img)
            
            # 转换为原始图片坐标
            img_center_x = int(coord_x * COORD_TO_IMAGE_SCALE)
            img_center_y = int(coord_y * COORD_TO_IMAGE_SCALE)
            img_radius = int(radius * COORD_TO_IMAGE_SCALE)
            
            # 绘制圆形轮廓（只有边框，无填充）
            bbox = [
                img_center_x - img_radius,
                img_center_y - img_radius,
                img_center_x + img_radius,
                img_center_y + img_radius
            ]
            draw.ellipse(bbox, outline=(255, 0, 0, 255), width=8)
            
            # 绘制中心点
            center_size = 10
            draw.ellipse(
                [img_center_x - center_size, img_center_y - center_size,
                 img_center_x + center_size, img_center_y + center_size],
                fill=(255, 0, 0, 255)
            )
        
        # 缩放图片
        new_width = int(img.width * self.zoom_scale)
        new_height = int(img.height * self.zoom_scale)
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 转换为 QPixmap
        img_bytes = img_resized.tobytes("raw", "RGBA")
        qimage = QImage(img_bytes, img_resized.width, img_resized.height, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimage)
        
        # 显示
        self.map_label.setPixmap(pixmap)
        self.map_label.adjustSize()
    
    def zoom_in(self):
        """放大"""
        if self.zoom_scale < 2.0:
            self.zoom_scale = min(2.0, self.zoom_scale + 0.1)
            self.update_display()
            self.update_zoom_label()
    
    def zoom_out(self):
        """缩小"""
        if self.zoom_scale > 0.2:
            self.zoom_scale = max(0.2, self.zoom_scale - 0.1)
            self.update_display()
            self.update_zoom_label()
    
    def update_zoom_label(self):
        """更新缩放显示"""
        if self.ring_data:
            coord_x, coord_y, radius = self.ring_data
            self.coord_label.setText(
                f"坐标: ({coord_x}, {coord_y}) | "
                f"毒圈等级: {self.current_ring_level} | "
                f"半径: {radius} | "
                f"缩放: {int(self.zoom_scale * 100)}%"
            )
        else:
            map_name = MAP_CONFIG[self.current_map]["name"] if self.current_map else ""
            self.coord_label.setText(
                f"地图: {map_name} | 点击地图以获取坐标 | "
                f"缩放: {int(self.zoom_scale * 100)}% | 滚轮可缩放"
            )
    
    def clear_ring(self):
        """清除毒圈"""
        self.ring_data = None
        self.update_display()
        self.update_zoom_label()


def main():
    """主函数"""
    app = QApplication(sys.argv)
    window = RingViewerWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
