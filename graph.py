import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle


def draw_enhanced_unet_with_pyramid():
    # 创建画布
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.axis('off')  # 关闭坐标轴

    # 设置显示范围，确保所有内容都在画布内
    ax.set_xlim(-3, 28)
    ax.set_ylim(0, 14)

    # 定义模块颜色
    colors = {
        'encoder': '#FF9999',
        'decoder': '#99CCFF',
        'attention': '#FFCC99',
        'pyramid': '#FFD700',
        'output': '#99FF99'
    }

    # 定义模块位置与标签
    modules = {
        # Encoder
        'Input': (0, 12, 'Input\n(3 x H x W)', colors['encoder']),
        'Conv1': (2, 12, 'Conv1\n(64 x H/2 x W/2)', colors['encoder']),
        'Down1': (4, 12, 'Down1\n(64 x H/4 x W/4)', colors['encoder']),
        'Down2': (6, 12, 'Down2\n(128 x H/8 x W/8)', colors['encoder']),
        'Down3': (8, 12, 'Down3\n(256 x H/16 x W/16)', colors['encoder']),
        'Down4': (10, 12, 'Down4\n(512 x H/32 x W/32)', colors['encoder']),

        # Pyramid Pooling (as a pyramid)
        'Pyramid1': (12, 9, '1x1', colors['pyramid']),
        'Pyramid2': (11.5, 7, '2x2', colors['pyramid']),
        'Pyramid3': (11, 5, '3x3', colors['pyramid']),
        'Pyramid4': (10.5, 3, '6x6', colors['pyramid']),
        'PyramidFusion': (13, 9, 'Fusion\n(512 x H/32 x W/32)', colors['pyramid']),

        # Attention Modules
        'ChannelAttention': (15, 9, 'Channel\nAttention', colors['attention']),
        'SpatialAttention': (17, 9, 'Spatial\nAttention', colors['attention']),

        # Decoder
        'Up1': (19, 12, 'Up1\n(256 x H/16 x W/16)', colors['decoder']),
        'Up2': (21, 12, 'Up2\n(128 x H/8 x W/8)', colors['decoder']),
        'Up3': (23, 12, 'Up3\n(64 x H/4 x W/4)', colors['decoder']),
        'Up4': (25, 12, 'Up4\n(64 x H/2 x W/2)', colors['decoder']),
        'Output': (27, 12, 'Output\n(n_classes x H x W)', colors['output']),
    }

    # 绘制模块矩形
    for name, (x, y, label, color) in modules.items():
        if "Pyramid" in name and name != "PyramidFusion":
            width = 1.2 - (12 - x) * 0.1  # 逐层缩小
            ax.add_patch(FancyBboxPatch(
                (x - width / 2, y), width, 1.2,
                boxstyle="round,pad=0.2", edgecolor='black', facecolor=color
            ))
        else:
            ax.add_patch(FancyBboxPatch(
                (x, y), 1.8, 1.2,
                boxstyle="round,pad=0.2", edgecolor='black', facecolor=color
            ))
        ax.text(x + 0.9, y + 0.6, label, ha='center', va='center', fontsize=10)

    # 定义数据流连接
    connections = [
        # Encoder connections
        ('Input', 'Conv1'), ('Conv1', 'Down1'), ('Down1', 'Down2'),
        ('Down2', 'Down3'), ('Down3', 'Down4'),

        # Pyramid Pooling connections
        ('Down4', 'Pyramid1'), ('Down4', 'Pyramid2'), ('Down4', 'Pyramid3'), ('Down4', 'Pyramid4'),
        ('Pyramid1', 'PyramidFusion'), ('Pyramid2', 'PyramidFusion'),
        ('Pyramid3', 'PyramidFusion'), ('Pyramid4', 'PyramidFusion'),

        # Attention connections
        ('PyramidFusion', 'ChannelAttention'), ('ChannelAttention', 'SpatialAttention'),

        # Decoder connections
        ('SpatialAttention', 'Up1'), ('Up1', 'Up2'), ('Up2', 'Up3'), ('Up3', 'Up4'), ('Up4', 'Output'),
    ]

    # 绘制箭头
    for src, dst in connections:
        x1, y1, _, _ = modules[src]
        x2, y2, _, _ = modules[dst]
        ax.annotate("", xy=(x2, y2 + 0.6), xytext=(x1 + 1.8, y1 + 0.6),
                    arrowprops=dict(arrowstyle=ArrowStyle("->", head_width=0.3), color='black'))

    # 显示标题
    plt.title("Enhanced U-Net with Pyramid Pooling and Attention Modules", fontsize=16)
    plt.show()


# 绘制详细的 U-Net 结构图
draw_enhanced_unet_with_pyramid()
