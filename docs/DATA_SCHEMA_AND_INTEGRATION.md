# TalentGraph 评测数据结构规范与前端对接指南 (V5 对比与最新契约)

> **文档定位**：本文档详细对比了老版本 V5 与最新版 `TalentGraph` 的数据结构差异，全面定义了最新标准 `ReportData` 契约规范（含 TypeScript 接口定义、字段级映射与前端组件渲染最佳实践），为未来新前端项目的无缝接入提供权威技术参考。

---

## 目录
1. [数据架构演进与设计原则](#一数据架构演进与设计原则)
2. [老版本 V5 vs 最新版 TalentGraph 结构对比](#二老版本-v5-vs-最新版-talentgraph-结构对比)
3. [最新版标准数据契约规范（TypeScript 定义）](#三最新版标准数据契约规范typescript-定义)
4. [字段级迁移与映射对照表](#四字段级迁移与映射对照表)
5. [前端对接与渲染最佳实践](#五前端对接与渲染最佳实践)

---

## 一、 数据架构演进与设计原则

| 维度 | 老版本 V5 数据模式 | 最新版 TalentGraph 契约模式 |
| :--- | :--- | :--- |
| **组织形式** | 散落在 10 多个子目录（`report_detail/`, `bullet_points/`, `labels/`, `result/*_plot/`） | **自包含统一 JSON** (`data/projects/{task_id}/report.json`) |
| **单/多候选人** | 按人切分为独立文件，前端无法直接获取团队全局大盘 | **项目级自闭环**：单份契约同时包含团队大盘与所有候选人完整详情 |
| **图表数据格式** | 服务端生成的静态 PNG 散点图/柱状图路径 | **ECharts 原生矢量结构**，前端可自由定制交互与动效 |
| **诊断与评语** | 16 个维度散碎的小评语（`summary.advantage`） | **结构化综合诊断画像**（总体评价、核心优势、薄弱项、成长建议、标签） |

---

## 二、 老版本 V5 vs 最新版 TalentGraph 结构对比

### 1. 老版本 V5 数据结构示例 (`report_detail/{member_number}.json`)
```json
{
  "overall": {
    "participant_total": 12,
    "tag": ["破局领航者", "逻辑严密"],
    "score": 88,
    "radar_image": "data/output/radar_plot/speaker_2.png",
    "radar_image_data": {
      "category": ["思维能力", "决策能力", "创新能力", "思维激发", "冲突协调", "推进能力"],
      "user_score": [85, 90, 82, 88, 79, 91],
      "avg_score": [75, 78, 74, 76, 73, 77]
    }
  },
  "detailAnalysis": [
    {
      "name": "思维能力 - 逻辑思维",
      "code": "1_1",
      "summary": {
        "advantage": ["能够在发言中准确由前置事实推导出后置决策举措"],
        "toImprove": ["在应对复杂多约束问题时的溯因推理仍有提升空间"]
      },
      "dimension_avg_score": 75,
      "user_dimension_score": 92,
      "images": ["data/output/1_1_plot/speaker_2.png"],
      "image_datas": []
    }
    // ...其余 15 个维度
  ]
}
```

---

### 2. 最新版 TalentGraph 标准数据结构示例 (`report.json`)
```json
{
  "task_id": "task_20260813_001",
  "project_id": "283607",
  "group_id": "271",
  "generated_at": "2026-08-13T06:30:00Z",
  "corpus_title": "智慧零售数字化转型战略研讨",
  "team_overview": {
    "participant_count": 12,
    "average_total_score": 78.5,
    "average_scores": {
      "1_1": 75.2,
      "1_2": 73.8,
      "1_3": 76.0,
      "2_1": 79.1,
      "2_2": 77.4,
      "2_3": 74.6,
      "3_1": 72.0,
      "3_2": 75.5,
      "3_3": 78.3,
      "4": 76.9,
      "5_1": 80.2,
      "5_2": 77.0,
      "5_3": 74.8,
      "6_1": 81.0,
      "6_2": 76.5,
      "6_3": 79.4
    },
    "collaboration_network": {
      "nodes": [
        { "id": "speaker_2", "name": "候选人 2", "symbolSize": 35, "category": 0 },
        { "id": "speaker_3", "name": "候选人 3", "symbolSize": 28, "category": 1 }
      ],
      "edges": [
        { "source": "speaker_2", "target": "speaker_3", "value": 3, "label": "支持" }
      ]
    }
  },
  "participants": [
    {
      "speaker_id": "speaker_2",
      "display_name": "候选人 2",
      "total_score": 88.5,
      "overall_rank": 1,
      "participant_total": 12,
      "speaking_word_count": 1420,
      "speaking_turns": 8,
      "matched_nodes_count": 9,
      "radar_data": [
        { "dimension_id": "1_1", "dimension_name": "逻辑思维", "category": "思维能力", "score": 92.0, "team_avg": 75.2 },
        { "dimension_id": "1_2", "dimension_name": "抽象归纳", "category": "思维能力", "score": 85.0, "team_avg": 73.8 },
        { "dimension_id": "1_3", "dimension_name": "系统思维", "category": "思维能力", "score": 88.0, "team_avg": 76.0 }
        // ...全量 16 个维度点位
      ],
      "dimension_details": {
        "1_1": {
          "dimension_id": "1_1",
          "dimension_name": "思维能力 - 逻辑思维",
          "category": "思维能力",
          "score": 92.0,
          "raw_score": 4.0,
          "rank": 1,
          "key_points": ["成功完成 4 次由前置事实到行动方案的演绎闭环推导"],
          "metrics": { "deduction_count": 3, "induction_count": 1, "abduction_count": 0 }
        }
        // ...其余 15 个维度的详细度量
      },
      "diagnosis": {
        "summary": "候选人2在讨论中展现出极强的战略推导能力与全局视野，多次在关键节点推进团队共识。",
        "strengths": [
          "逻辑严密：能够从复杂的市场事实中精准推演出具体的执行举措",
          "推进高效：在讨论后半程展现出强大的共识收敛能力"
        ],
        "weaknesses": [
          "跨界联想：在面对非本专业领域的创新发散方案时参与度略显不足"
        ],
        "growth_suggestions": [
          "在破局阶段可更加大胆地提出跨行业借鉴方案",
          "适度增加对分歧观点的正面回应与调和"
        ],
        "tags": ["破局领航者", "逻辑严密", "战略收敛专家"]
      }
    }
  ]
}
```

---

## 三、 最新版标准数据契约规范（TypeScript 定义）

前端可以直接复制以下 TypeScript 强类型接口进行接入与类型校验：

```typescript
/** 1. 16 维雷达图数据点 */
export interface RadarPoint {
  dimension_id: string;      // 维度编号: "1_1" ~ "6_3"
  dimension_name: string;    // 维度短名: "逻辑思维"
  category: string;          // 能力大类: "思维能力" | "决策能力" | "创新能力" | "思维激发" | "冲突协调" | "推进能力"
  score: number;             // 候选人标准得分 (60 ~ 98)
  team_avg: number;          // 团队该项平均分
}

/** 2. 单维度细分得分与度量详情 */
export interface DimensionDetail {
  dimension_id: string;      // "1_1"
  dimension_name: string;    // "思维能力 - 逻辑思维"
  category: string;          // "思维能力"
  score: number;             // 最终得分 (0 ~ 100)
  raw_score: number;         // 原始数学/图论指标
  rank: number;              // 单项在全场的排名 (1 ~ N)
  key_points: string[];      // 行为佐证要点
  metrics: Record<string, any>; // 细分算法指标 (如推理次数、中心性值等)
}

/** 3. 候选人全维专家综合诊断 */
export interface ComprehensiveDiagnosis {
  summary: string;           // 全场整体表现综合评述
  strengths: string[];       // 核心优势与高光行为 (2~3条)
  weaknesses: string[];      // 主要薄弱环节与待改进项 (1~2条)
  growth_suggestions: string[]; // 针对性成长进阶建议 (2~3条)
  tags: string[];            // 核心人才个性化标签 (如 ["破局领航者", "逻辑严密"])
}

/** 4. 单个候选人全量报告模型 */
export interface ParticipantReport {
  speaker_id: string;        // 标识符: "speaker_2"
  display_name: string;      // 显示名称: "候选人 2"
  total_score: number;       // 综合加权总分 (0 ~ 100)
  overall_rank: number;      // 全场综合总排名 (1 ~ N)
  participant_total: number; // 全场总参评人数
  speaking_word_count: number; // 发言总字数
  speaking_turns: number;    // 发言总轮次
  matched_nodes_count: number; // 命中核心图谱节点数
  radar_data: RadarPoint[];  // 16 维雷达图序列
  dimension_details: Record<string, DimensionDetail>; // 16 维细分字典
  diagnosis: ComprehensiveDiagnosis; // 综合诊断画像
}

/** 5. 团队互动网络图节点与边 (ECharts Graph 拓扑) */
export interface NetworkNode {
  id: string;
  name: string;
  symbolSize: number;
  category?: number;
}

export interface NetworkEdge {
  source: string;
  target: string;
  value: number;
  label?: string;
}

export interface CollaborationNetwork {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
}

/** 6. 团队大盘宏观数据 */
export interface TeamOverview {
  participant_count: number;
  average_total_score: number;
  average_scores: Record<string, number>; // 16 维度团队均分
  collaboration_network: CollaborationNetwork;
}

/** ⭐ 7. 全局顶层 ReportData 契约 */
export interface ReportData {
  task_id: string;           // 任务 ID
  project_id: string;        // 所属项目/会议 ID
  group_id: string;          // 小组 ID
  generated_at: string;      // 生成时间 (ISO8601)
  corpus_title: string;      // 题库案例名称
  team_overview: TeamOverview; // 团队大盘
  participants: ParticipantReport[]; // 全员报告列表 (默认已按总排名升序排列)
}
```

---

## 四、 字段级迁移与映射对照表

| 业务含义 | 老版本 V5 字段路径 | 最新版 TalentGraph 字段路径 | 说明与优势 |
| :--- | :--- | :--- | :--- |
| **参评总人数** | `overall.participant_total` | `team_overview.participant_count` 或 `participant.participant_total` | 全局大盘与个人对象均可直接获取 |
| **候选人总分** | `overall.score` (整数) | `participant.total_score` (浮点数) | 保留小数精度，排序更精细 |
| **人才标签** | `overall.tag` (数组) | `participant.diagnosis.tags` (数组) | 归纳在结构化诊断画像中 |
| **综合总评语** | 无（老版散碎在 16 个小文件） | `participant.diagnosis.summary` (字符串) | 具备全局视角的长篇综合诊断 |
| **核心优势** | `detailAnalysis[i].summary.advantage` | `participant.diagnosis.strengths` (数组) | 提炼全场核心亮点，不再拆碎在每个单维度 |
| **薄弱环节** | `detailAnalysis[i].summary.toImprove` | `participant.diagnosis.weaknesses` (数组) | 提炼关键短板 |
| **成长建议** | 无 | `participant.diagnosis.growth_suggestions` (数组) | 专为人才发展规划设计的新增字段 |
| **16 维雷达图** | `overall.radar_image_data` (6大类) | `participant.radar_data` (16个细分点位) | 直接支持 16 维深度对比与大类聚合 |
| **单维平均分** | `detailAnalysis[i].dimension_avg_score` | `radar_data[i].team_avg` 或 `team_overview.average_scores["1_1"]` | 获取方式更灵活统一 |
| **单维个人分** | `detailAnalysis[i].user_dimension_score` | `radar_data[i].score` 或 `dimension_details["1_1"].score` | 类型由 `number` 统一承接 |
| **团队互动拓扑** | 静态生成的图片路径 | `team_overview.collaboration_network` | 原生支持 ECharts 力导向互动拓扑渲染 |

---

## 五、 前端对接与渲染最佳实践

### 1. API 接口获取
前端仅需发起 **1 次 HTTP 请求** 即可拉取完整评测大盘与所有候选人数据：

```typescript
// services/api.ts
export async function fetchAssessmentReport(taskId: string): Promise<ReportData> {
  const response = await fetch(`http://localhost:8000/api/v1/assessment/${taskId}/data`);
  if (!response.ok) {
    throw new Error(`Failed to load assessment data: ${response.statusText}`);
  }
  return await response.json();
}
```

### 2. WebSocket 实时进度监听
在提交评测后，前端可建立 WebSocket 连接实时展示 0% ~ 100% 进度条与当前流水线阶段：

```typescript
// hooks/useAssessmentProgress.ts
export function useAssessmentProgress(taskId: string, onUpdate: (progress: number, stage: string) => void) {
  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8000/api/v1/ws/progress/${taskId}`);
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      onUpdate(data.progress, data.stage_description || data.stage);
    };
    return () => ws.close();
  }, [taskId]);
}
```

### 3. ECharts 16 维雷达图配置参考
```typescript
function buildRadarOption(radarData: RadarPoint[], displayName: string) {
  return {
    title: { text: `${displayName} 16 维能力模型评估` },
    tooltip: { trigger: 'item' },
    legend: { data: ['个人得分', '团队均分'] },
    radar: {
      indicator: radarData.map(p => ({ name: p.dimension_name, max: 100 })),
      splitNumber: 4,
    },
    series: [{
      type: 'radar',
      data: [
        {
          value: radarData.map(p => p.score),
          name: '个人得分',
          areaStyle: { opacity: 0.25, color: '#3b82f6' },
          lineStyle: { color: '#3b82f6', width: 2 }
        },
        {
          value: radarData.map(p => p.team_avg),
          name: '团队均分',
          lineStyle: { type: 'dashed', color: '#94a3b8', width: 1.5 }
        }
      ]
    }]
  };
}
```

### 4. 候选人总榜与画像卡片渲染示例 (Vue / React 伪代码)
```html
<!-- 候选人总榜 -->
<div class="ranking-table">
  <table>
    <thead>
      <tr><th>排名</th><th>候选人</th><th>总分</th><th>核心标签</th><th>发言轮次</th></tr>
    </thead>
    <tbody>
      <tr v-for="p in report.participants" :key="p.speaker_id">
        <td>第 {{ p.overall_rank }} 名</td>
        <td>{{ p.display_name }}</td>
        <td>{{ p.total_score.toFixed(1) }}</td>
        <td>
          <span v-for="tag in p.diagnosis.tags" class="badge">{{ tag }}</span>
        </td>
        <td>{{ p.speaking_turns }} 轮</td>
      </tr>
    </tbody>
  </table>
</div>
```
