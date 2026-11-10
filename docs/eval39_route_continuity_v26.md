# v26: 修复 eval-39 下楼后误回头（已实现）

## 实际根因和限定范围

来自新上传的 `eval-batch4-39.rar`：动作 17–22 沿 255° 连续执行六次、RGB-D 均变化，随后到站位 `0:6` 的墙/转角；原 Prompt/叙述把新的前方 BLOCKED 误当作整个先前下楼路线失败。动作 23–81 中大量原地转向，后来朝 90° 前进回到视觉上较高的楼梯。导航决策 `obs_0084` 附带三个同一站位、同一 yaw/pitch 的高度重复历史 RGB，关键 `obs_0019/obs_0025` 不在图像列表。模型还曾把 90°->210° 的历史方位误称 RIGHT 120°，执行器并无转向符号错误。

**已改：**

1. **只读事实关联** `havln_agent/navigation_evidence.py::_endpoint_obstacle` 在已完成前进段的真实终点及航向、正确的观测时间顺序上，查找后续被观测或真正拒绝的前方障碍。把 `FRONT_BLOCK_OBSERVED_NO_FORWARD_ATTEMPT` 与 `NEXT_FORWARD_REJECTED` 分开输出到已存在的 `route_continuity.endpoint_front_obstacle`，明确不能否定已经执行且 RGB-D 发生变化的先前动作。不推断真正位移、楼梯下降或“下楼完成”。保持 Memory 唯一事实源，不持久化第二份摘要。
2. **真实历史 RGB 配额与去重** `havln_agent/navigation_history.py::navigation_visual_anchors`：当当前方向与重要前进段接近相反（>=105°）或站位与其终点一致，先保证该段起点和终点图像的候选优先权，再考虑最近的真实前进图像、ObservationAgent 候选、最新的重访图及其他多样化画面。相同执行前进站位、近似相同 yaw/pitch 只留一个代表，避免 RGB 哈希变化导致假“新视角”。当前 RGB 始终是动态人物、实时几何的准确信息。NavigationAgent 提高历史 RGB 上限为 **5**，且不会无条件补满；当前 RGB 加历史 RGB 最多 6 张。所有图均需真实 RGB-D hash 验证。ObservationAgent 使用同一个历史关键帧选择，维持最多 3 对 RGB-D。
3. **相对方向清晰标记** `mission_history.py` 对同一执行前进站位、具有可信绝对目标 yaw 的已有路径候选，确定性计算 `relative_side_from_current` 和 `relative_turn_deg_from_current`；不在模型执行接口恢复 `target_heading_deg`。跨站位没有可执行方向，精简输入不再给模型候选的 `absolute_heading_deg`，避免模型把历史方向误算成当前位置 RIGHT/LEFT。绝对 yaw 仍存在原始 Memory，作为带来源的历史视角标签。
4. **选择性委托 VLM + 不空转** NavigationAgent 提示：遇到“这楼梯是向上还是向下”等具体歧义且会改变路线时，调用 `observe_for_goal` 并引用历史观测 ID；清楚看到单一可探索的连续通路、且当前未确认 BLOCKED / 无显著人物与落差风险时，优先考虑小段自主前进，而不是继续无目的 TURN/LOOK；底层每原子 Depth 仍独立检查。ObservationAgent 只负责图像和建议，不授权前进。
5. **输入精简** 继续用已有 `mission_evidence_v1` / `route_continuity` 与 v23 聚合；终点阻挡与该端点已携带的重复单独观察记录在导航精简上下文中去重。没有引入新的 sg0/sg1、语义验证器、最大原子步数锁、强制前进或禁止回头规则。

## 针对 eval-39 的离线回归

`tests/test_v26_eval39_route_quality.py` 用真实轨迹的观测 ID/动作方向构造一个不依赖 Habitat 资源的可复现最小 Memory：`obs_0019 -> obs_0025` / 17–22 步，当前 `obs_0084` 航向 90°，同站位重复 `obs_0044/0048/0078`，终点 `obs_0025` 的后来前向障碍。断言历史前后图优先、重复角度不占满、该阻挡与已执行的六次动作因果顺序正确、来源错误时不关联，以及同站位候选从 90° 到 210° 为 LEFT 120° 而跨站位没有直接可执行方向。

```bash
python -m compileall -q havln_agent
python -m unittest discover -s tests -p 'test_v26_eval39_route_quality.py' -v
python -m unittest discover -s tests -p 'test_v23_route_evidence.py' -v
python -m unittest discover -s tests -p 'test_v22_active_pitch_heading.py' -v
```

下一次同一 val_seen Episode/100 步实测应对比 v25 的 12 次前进、78 次转向及回走现象，重点看是否利用 `obs_0019/0025` 纠正路线、是否避免连续多圈无信息观察、是否进入 foyer。不要以单次 MOVE 数量认定成功；不要仅凭 RGB-D hash 变化断言实际位移/高度变化。行人安全修复仍在以后独立处理，不随本 PR 改动。
