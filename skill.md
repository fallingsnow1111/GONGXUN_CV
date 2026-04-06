# gongxun Vision 开发进度记录

更新时间：2026-04-06

## 状态标签说明
- [SOLVED]：已定位并完成修复，当前版本稳定复现通过。
- [MONITORING]：已有临时策略，仍需持续观察或现场回归。
- [OPEN]：问题已记录，尚未进入有效修复阶段。

## 硬件环境快照（当前）
| 设备 | 角色 | 连接方式 | 关键参数 |
| :--- | :--- | :--- | :--- |
| HF901（现场主摄像头） | 主摄像头 | USB（Windows） | 常用 640x480@30；索引以 find_camera_id.py 实测为准 |
| PC（Windows） | 开发主机 | 本地运行 | Python + OpenCV，当前以 print 代替 serial 联调 |
| Raspberry Pi 5（目标部署） | 比赛部署主机 | 本地运行 | Cortex-A76，目标承载 CV+Kalman+YOLO 混合链路 |
| 下位机串口链路 | 协议联调对象 | 串口（预留） | 协议帧 55 5A action x y AA |

## 0. 开发流程约定（新增）
- 每次代码改动完成后，必须同步更新本文件。
- 每次更新至少包含：
  - 本次改动内容
  - 踩到的坑
  - 解决方法
  - 验证结果（是否通过）
- 若存在未解决问题，必须追加到“阻塞项/待验证”中。
- 新增记录筛选规则：仅代码整理、路径修改、文档格式调整这类与“方案优化/bug 修复”无关的改动，不写入本文件。

## 1. 目标
- 圆盘机场景下实现红/绿/蓝色块识别与圆环识别。
- 在无串口设备阶段，先完成视觉链路与协议帧输出验证。
- 引入卡尔曼滤波降低中心点抖动与短时丢检影响。

## 2. 当前已完成
- 已完成按平台拆分目录（不改代码内容）：
  - windows/: color_line_det_windows.py, disk_color_ring_kalman.py, find_camera_id.py, test_windows.py, hsv_snapshot_picker.py, threshold_tuner.py
  - raspberry_pi/: color_line_det.py, track.py
- 新增独立主脚本：windows/disk_color_ring_kalman.py
- 支持模式（键盘 0-6）：
  - 1 红色块
  - 2 绿色块
  - 3 蓝色块
  - 4 红色环
  - 5 绿色环
  - 6 蓝色环
- 识别流程已打通：
  - 预处理（GaussianBlur）
  - HSV 阈值分割（红色双区间）
  - 形态学处理
  - 轮廓筛选（面积+圆度）
  - 中心点提取
  - 卡尔曼滤波输出
- 圆环检测已加入内孔/填充率约束，降低把实心色块误判为圆环。
- 输出改为打印协议帧（默认 print 模式），格式保持兼容：
  - 55 5A action x y AA
- 已新增摄像头探测工具：windows/find_camera_id.py
- 已确认外接 USB 摄像头索引：2（历史现场也出现过 index=0，以实测为准）
- 主脚本已固定摄像头策略：
  - PREFERRED_CAMERA_INDICES = [2]
  - ALLOW_INDEX_0_FALLBACK = False
- 相机初始化已回退为稳定方案（方案一）：
  - 固定 640x480 @ 15 FPS
  - 不强制 MJPG/FourCC
  - 参数设置后预热 0.5 秒再读首帧
  - 主后端失败时仅轻量回退 CAP_ANY
- 新增 Windows 兼容脚本：color_line_det_windows.py（不改动原 color_line_det.py）
  - 默认外接摄像头 index=2
  - 默认打印协议帧（55 5A action x y AA）
  - 保留原有颜色/圆环/直线模式逻辑
  - 支持键盘 0-9 切模式，q 退出
  - Linux/Raspberry Pi 分支新增稳定打开策略：
    - 扫描 /dev/videoX（优先 /dev/video2）
    - 强制 MJPG
    - 固定 640x480 @ 30 FPS
    - frame.mean > 10 作为有效帧验收
  - 现已补全显式色环识别流程（模式 4/5/6）：
    - 使用 RETR_CCOMP 获取轮廓层级
    - 结合 circularity + fill_ratio + has_hole 判定环形目标
    - 独立于色块模式（1/2/3/7/8）

## 3. 当前运行方式
- 探测摄像头索引：
  - python windows/find_camera_id.py
- 抓拍+滴管取值（建立 HSV 粗范围）：
  - python windows/hsv_snapshot_picker.py --camera-index 0
- 启动主识别：
  - python windows/disk_color_ring_kalman.py
- 启动 Windows 兼容旧逻辑版本：
  - python windows/color_line_det_windows.py
- 实时 Trackbar 调阈值（在线微调）：
  - python windows/threshold_tuner.py --camera-index 0
  - python windows/threshold_tuner.py --image windows/debug_samples/sample_xxx.jpg
- 启动 Raspberry Pi 原版脚本：
  - python raspberry_pi/color_line_det.py
- 运行中快捷键：
  - 0 关闭识别
  - 1~6 切换对应色块/圆环模式
  - q 退出

## 4. 已知配置与约束
- 当前输出模式：print（无串口硬件阶段）
- 串口模式保留：将 OUTPUT_MODE 改为 serial 可恢复真实发送
- 坐标缩放系数：
  - SCALE_X = 3.5
  - SCALE_Y = 2.5
- 摄像头目标分辨率与帧率：640x480, 30 FPS

## 5. 本轮踩坑与解决（2026-03-30，已精简）
- [SOLVED] 坑 1：无串口设备阶段无法验证输出链路。
  - 现象：识别逻辑运行但无法看到协议发送结果。
  - 解决：发送链路改为打印协议帧（55 5A action x y AA），保留 serial 模式开关。
- [SOLVED] 坑 2：用户不希望继续改原文件，避免反复撤销影响节奏。
  - 现象：对原文件直接改动容易与现场验证版本冲突。
  - 解决：改为新建独立文件 color_line_det_windows.py，原文件保持不动。
- [SOLVED] 坑 3：用户需要对外讲解代码，但色环流程与色块流程混在一起不直观。
  - 现象：模式 4/5/6 看起来像普通色块阈值，不利于讲解和后续调参。
  - 根因：没有独立的色环检测函数和独立分支。
  - 解决：新增 find_ring 与独立模式分支，显式区分色块识别和色环识别。
- [SOLVED] 坑 4：代码防御逻辑过重，调试与讲解成本高。
  - 现象：相机探测和检测流程分支过多，讲解时主线不清晰。
  - 根因：早期为兼容复杂现场加入了多层兜底。
  - 解决：完成瘦身重构：
    - 摄像头打开改为固定 index+后端（Windows: DSHOW，失败再 CAP_ANY）
    - get_color_mask 简化为统一边界列表合并
    - detect_target 统一色块/色环检测逻辑，减少重复代码
    - 模式分发改为 MODE_CONFIG 表驱动
- [MONITORING] 坑 5：红/绿环识别距离敏感，需靠很近才能稳定识别。
  - 现象：红环和绿环在中远距离下检出率偏低，绿环更明显。
  - 根因：疑似 HSV 阈值与环形判定参数（面积/填充率/圆度）在现场光照下偏紧。
  - 解决：本轮先记录问题与目录整理，不改算法；下一轮优先做阈值重标定与环形参数联调。
- [MONITORING] 坑 6：三色环整体都需要近距离，且绿色环显示与识别最不明显。
  - 现象：红/绿/蓝三种环在稍远距离都不稳定，绿色环边缘最容易发虚或断裂。
  - 根因：细环在当前分辨率和光照下像素占比低，叠加阈值与形态学后有效连通区域不足。
  - 解决：本轮仅记录现象，不改代码；后续优先按“绿环→红环→蓝环”顺序单独重标阈值并调整环形最小面积/圆度参数。

## 6. 本次改动记录（新增）
- 改动内容：
  - 新增阈值调试工具脚本：
    - hsv_snapshot_picker.py（按 s 抓拍静态样本 + 鼠标点击读取 HSV）
    - threshold_tuner.py（6 通道 HSV Trackbar + 模糊/腐蚀/膨胀联调）
  - 新增参数导出与证据保存能力：
    - threshold_tuner.py 支持按 p 打印当前阈值，按 s 保存 frame/mask/json。
  - 按评审建议优化 skill.md 结构：新增状态标签、硬件环境快照、量化验证模板、讲解话术与截图索引。
  - 给坑位补充状态标签（SOLVED / MONITORING），提升后续检索效率。
  - 本轮改动仅新增调试脚本，未改主识别逻辑。
- 解决方法：
  - 采用“三步走”流程落地：
    - 第一步：抓拍固定光照样本，避免动态盲调。
    - 第二步：鼠标滴管读取多点 HSV，自动给出建议 lower/upper。
    - 第三步：Trackbar 实时微调并保存可复用阈值配置。
  - 文档层：将“现象 -> 根因 -> 解决”保留不变，仅增强索引能力与可量化表达。
  - 管理层：将仍需持续回归的问题标记为 [MONITORING]，避免与已闭环问题混淆。
- 验证结果：
  - hsv_snapshot_picker.py 静态检查通过（No errors found）。
  - threshold_tuner.py 静态检查通过（No errors found）。
  - 两个脚本均已支持 Windows 摄像头索引参数与调参结果落盘。
  - 文档内已能快速区分“已解决问题”和“持续观察问题”。

## 7. 阻塞项/待验证
- 红/绿环中远距离识别率偏低，需从“颜色域主导”升级为“几何域主导”，避免仅靠 HSV 调参。
- 蓝环在中距离同样存在检出不稳定，需统一按细环场景做参数重标与几何约束联调。
- [OPEN] YOLO（物料）与圆环检测链路当前仅并行存在，缺少时间对齐、目标级关联与置信度融合机制。

## 8. 待办（下一阶段，按 P0/P1/P2）
- P0（必须做）
  - 引入“边缘 + 圆检测”主链路（方案 B：gray/blur/Canny/findContours/fitEllipse 或 minEnclosingCircle），用于替代纯 HSV 主导的圆环检测。
  - 升级 Kalman：状态向量改为 [cx, cy, vx, vy]，补齐丢检预测与 gating 机制。
  - 增加 ROI 动态机制：有历史目标时局部搜索，无目标时全图搜索。
- P1（强烈建议）
  - 建立小规模数据集与批量评估流程：dataset/green_ring/{near,mid,far}，后续扩展到 red_ring、blue_ring。
  - 环结构参数升级：引入 R_outer、R_inner 与 ring_thickness_ratio，降低 fill_ratio 对距离变化的敏感性。
  - 增加滤波前后抖动统计（标准差）与误检率统计，形成“改进前/后”可比报告。
- P2（优化项）
  - 在树莓派 5 上执行 YOLO + OpenVINO 压测（输入 320/416/640），输出 FPS、延迟、CPU 占用对照表。
  - 多目标关联机制预研：tracking ID + data association（IoU/Hungarian）。
  - 部署模式关闭 imshow，减少 GUI 渲染开销；若后续接回串口再联调 action 语义与坐标缩放标定。

## 9. 风险与注意
- 不同光照下 HSV 可能漂移，优先先调阈值再调卡尔曼噪声。
- 目前按单目标优先策略处理；若同色多目标并存，需要加入目标关联机制。
- 树莓派 5 仅靠 CPU 跑 YOLO 时，稳态 30 FPS 压力较大；需依赖异步+输入降采样+OpenVINO 才有机会逼近目标。

## 10. 量化验证基线（模板）
- 验证场景：近距离（20-30cm）/中距离（40-60cm）/远距离（70cm+）
- 关键指标：
- detection_rate（10 秒内有效帧占比）：N/A（待补）
- false_positive_rate（10 秒内误检帧占比）：N/A（待补）
- jitter_std（中心点像素标准差，滤波前/后各一组）：N/A（待补）
- latency_ms（采集到发送的端到端时延）：N/A（待补）
- cpu_usage（分辨率/帧率对应的 CPU 占用）：N/A（待补）
- 当前已确认：
  - 协议输出链路可工作（print 模式，帧格式 55 5A action x y AA）
  - 细环场景下三色环中距离检出仍需专项优化

## 11. 圆环距离敏感对外讲解话术（可直接复用）
- 由于绿色在 HSV 空间中与部分背景/反光区域重合度更高，且环形目标本身线宽较细，远距离时可用轮廓像素会明显下降，导致面积特征更容易低于阈值下限。
- 当前策略属于“保准不保全”：优先保证近距离识别准确率，再通过阈值与环形参数联调逐步提升中远距离召回。

## 12. 证据截图索引（建议）
| 日期 | 模式 | 距离 | 截图路径 | 观察结论 |
| :--- | :--- | :--- | :--- | :--- |
| 待补 | 5 绿色环 | 中距离 | 待补（建议保存 mask 与 result 同帧） | 绿色环边缘断裂、检出间歇 |

## 13. 最终预期实现方案（2026-04-02）
- 总体目标：
  - 在树莓派 5 上实现“CV + 卡尔曼（圆环）+ YOLO（物料）”混合方案，优先追求稳定与实时并重，冲击接近 30 FPS 的比赛可用状态。
- 架构策略：
  - 主循环（高频）：负责采集图像 + HSV 圆环检测 + 卡尔曼滤波 + 协议输出。
  - YOLO 线程（低频）：独立执行物料识别，目标频率 10-15 FPS，通过共享缓存发布最新识别结果。
  - 解耦原则：主循环不等待 YOLO 完成；使用“最新可用结果”机制避免阻塞。
- 推理部署优先级：
  - 首选 OpenVINO（CPU 优化），模型建议 YOLO Nano 级别。
  - 输入尺寸按 320/416/640 进行压测，优先选择可稳定满足时延预算的配置。
  - 若后续具备 NPU（如 Hailo），再升级到更高吞吐方案。
- 识别策略（按算力分层）：
  - 方案 A（算力充足）：物料与圆环均由 YOLO 承担，CV 作为兜底或校验。
  - 方案 B（当前默认）：物料用 YOLO，圆环继续使用 HSV + 卡尔曼滤波。
- 实战优化要点：
  - 部署模式关闭 imshow。
  - 形态学参数优先“更大核 + 更少迭代”，避免 CPU 过载。
  - 统一验证近/中/远距离下的检出率、抖动、时延与 CPU 占用。

## 14. 本次改动记录（2026-04-02）
- 改动内容：
  - 删除“找不到摄像头/画面黑帧”相关坑位记录，保留与比赛主线直接相关的问题与经验。
  - 新增“最终预期实现方案（2026-04-02）”，明确树莓派 5 上的混合方案路径与算力分层策略。
  - 更新待办与风险项，补充异步 YOLO、OpenVINO 与部署模式性能优化方向。
- 踩到的坑：
  - 历史坑位记录过于偏向设备接入细节，文档主线被稀释，影响赛题方案沟通效率。
- 解决方法：
  - 对坑位列表做主题化精简，仅保留输出链路、架构可维护性、识别效果相关记录。
  - 将“方案预期”单独成节，明确 A/B 两套可落地路径，避免开发阶段反复摇摆。
- 验证结果：
  - 文档已完成主线收敛：从“设备问题复盘”转为“比赛落地方案+性能目标”。
  - 当前记录已满足“本次改动/踩坑/解决方法/验证结果”四要素。

## 15. 本次改动记录（2026-04-02，可视化量化指标与环形方框）
- 改动内容：
  - 更新 [windows/test_windows.py](windows/test_windows.py)：新增实时可视化量化指标叠层，包含 `FPS`、`DetectRate(2s)`、`Mask占比`、`面积(area)`、`圆度(circularity)`、`填充率(fill_ratio)` 与中心坐标。
  - 新增明显识别状态提示：`DETECTED / NOT DETECTED`，用于快速判断“当前是否识别到目标”。
  - 在色环识别分支（模式 4/5/6 与模式 9 的绿环）新增矩形方框标注（`cv.rectangle`），与轮廓/外接圆同时显示。
  - 修正 `COLOR_THRESHOLDS` 中红色阈值结构，确保双区间配置在 `red` 节点下生效。
- 踩到的坑：
  - 初次补丁时在 `process_frame` 末尾引入了缩进错误，导致静态检查报“意外缩进”。
- 解决方法：
  - 重新整理 `process_frame` 末尾结构，将 `detected/metric_target` 对齐到目标检测分支层级。
  - 将 `cv.imshow("mask", ...)`、运行指标更新与 HUD 绘制统一放到分支外，确保 0/1-9 全模式都能显示状态与指标。
- 验证结果：
  - [windows/test_windows.py](windows/test_windows.py) 静态检查通过（No errors found）。
  - 当前未接入现场实机回归，本次验证结论为“代码层通过，实机效果待你现场确认”。

## 16. 本次改动记录（2026-04-06，摄像头打开失败排查）
- 改动内容：
  - 参考 [windows/test_windows.py](windows/test_windows.py) 的相机初始化策略，升级 [windows/hsv_snapshot_picker.py](windows/hsv_snapshot_picker.py) 与 [windows/threshold_tuner.py](windows/threshold_tuner.py) 的 `open_camera`：
    - Windows 端优先 `CAP_DSHOW`，失败回退 `CAP_ANY`。
    - 设置 `MJPG` FourCC（仅 Windows）。
    - 设置分辨率/FPS 后增加预热（0.5s）与首帧有效性校验（最多重试 10 次）。
  - 新增索引回退机制：若指定索引失败，自动尝试 0~5 其余索引，成功后打印实际使用索引。
  - 新增更明确的失败提示：打开失败时提示先运行 [windows/find_camera_id.py](windows/find_camera_id.py)。
- 踩到的坑：
  - 原脚本仅做“单索引 + 简单打开”判断，现场当索引变化或首帧无效时会直接报“Camera open failed”。
  - 无日志输出实际后端、实际分辨率与实际 FPS，定位成本高。
- 解决方法：
  - 对齐主测试脚本的稳定策略（后端回退 + MJPG + 预热 + 首帧校验）。
  - 增加索引自动回退与打开成功日志（index/backend/size/fps/frame_shape）。
  - 在失败路径补充明确排障入口（`find_camera_id.py`）。
- 验证结果：
  - [windows/hsv_snapshot_picker.py](windows/hsv_snapshot_picker.py) 静态检查通过（No errors found）。
  - [windows/threshold_tuner.py](windows/threshold_tuner.py) 静态检查通过（No errors found）。
  - 尚未完成实机回归；需在现场执行并确认最终选中的 camera index 与画面稳定性。

