from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, Iterable, Tuple


class CoordinateSystem(str, Enum):
    POLAR = "polar"
    CARTESIAN = "cartesian"


class ChartCapability(str, Enum):
    TYPE_DETECTION = "type_detection"
    GRID_ENCRYPTION = "grid_encryption"
    EVALUATION = "evaluation"


@dataclass(frozen=True)
class ChartDefinition:
    chart_type: str
    label: str
    description: str
    coordinate_system: CoordinateSystem
    capabilities: FrozenSet[ChartCapability]


DEFAULT_CHART_TYPE = "v_bar"

_COMMON_CAPABILITIES = frozenset(
    {
        ChartCapability.TYPE_DETECTION,
        ChartCapability.GRID_ENCRYPTION,
        ChartCapability.EVALUATION,
    }
)

CHART_DEFINITIONS: Tuple[ChartDefinition, ...] = (
    ChartDefinition(
        "rose",
        "玫瑰图",
        "南丁格尔玫瑰图，多个扇区从中心向外延伸，角度通常均匀分布",
        CoordinateSystem.POLAR,
        _COMMON_CAPABILITIES,
    ),
    ChartDefinition(
        "radar",
        "雷达图",
        "多个坐标轴从中心向外辐射，形成多边形或雷达网格",
        CoordinateSystem.POLAR,
        _COMMON_CAPABILITIES,
    ),
    ChartDefinition(
        "v_bar",
        "垂直条形图",
        "条形垂直排列，通常用于比较类别数据",
        CoordinateSystem.CARTESIAN,
        _COMMON_CAPABILITIES,
    ),
    ChartDefinition(
        "h_bar",
        "水平条形图",
        "条形水平排列，通常用于比较类别数据",
        CoordinateSystem.CARTESIAN,
        _COMMON_CAPABILITIES,
    ),
    ChartDefinition(
        "line",
        "折线图",
        "数据点通过直线连接，显示趋势变化",
        CoordinateSystem.CARTESIAN,
        _COMMON_CAPABILITIES,
    ),
    ChartDefinition(
        "scatter",
        "散点图",
        "数据点分布在直角坐标系中，用于显示两个变量之间的关系",
        CoordinateSystem.CARTESIAN,
        _COMMON_CAPABILITIES,
    ),
    ChartDefinition(
        "bubble",
        "气泡图",
        "散点图变体，气泡大小通常表示第三个变量",
        CoordinateSystem.CARTESIAN,
        _COMMON_CAPABILITIES,
    ),
    ChartDefinition(
        "donut",
        "环形图",
        "中心有空洞的环形占比图",
        CoordinateSystem.POLAR,
        _COMMON_CAPABILITIES,
    ),
    ChartDefinition(
        "pie",
        "饼图",
        "圆形被分割成多个扇区，表示各部分占比",
        CoordinateSystem.POLAR,
        _COMMON_CAPABILITIES,
    ),
)

CHARTS_BY_TYPE: Dict[str, ChartDefinition] = {
    definition.chart_type: definition for definition in CHART_DEFINITIONS
}

SUPPORTED_CHART_TYPES = tuple(CHARTS_BY_TYPE)
POLAR_CHART_TYPES = frozenset(
    chart_type
    for chart_type, definition in CHARTS_BY_TYPE.items()
    if definition.coordinate_system == CoordinateSystem.POLAR
)
CARTESIAN_CHART_TYPES = frozenset(
    chart_type
    for chart_type, definition in CHARTS_BY_TYPE.items()
    if definition.coordinate_system == CoordinateSystem.CARTESIAN
)


def get_chart_definition(chart_type: str) -> ChartDefinition:
    return CHARTS_BY_TYPE.get(chart_type, CHARTS_BY_TYPE[DEFAULT_CHART_TYPE])


def normalize_chart_type(chart_type: str) -> str:
    return chart_type if chart_type in CHARTS_BY_TYPE else DEFAULT_CHART_TYPE


def get_coordinate_system(chart_type: str) -> CoordinateSystem:
    return get_chart_definition(chart_type).coordinate_system


def supports_capability(chart_type: str, capability: ChartCapability) -> bool:
    return capability in get_chart_definition(chart_type).capabilities


def format_chart_options(definitions: Iterable[ChartDefinition] = CHART_DEFINITIONS) -> str:
    return "\n".join(
        f"- {definition.chart_type}: {definition.label}，{definition.description}"
        for definition in definitions
    )


def format_supported_types(types: Iterable[str] = SUPPORTED_CHART_TYPES) -> str:
    return "、".join(f'"{chart_type}"' for chart_type in types)
