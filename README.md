# Squat Coach Prototype

基于浏览器摄像头和 MediaPipe Pose 的深蹲 AI 教练原型，用于向老师演示“特定力量训练动作实时反馈”的核心闭环。

## 启动

```powershell
cd C:\Users\ocean\squat-ai-demo
npm install
npm start
```

然后打开：

```text
http://localhost:8000
```

如需直接进入演示模式：

```text
http://localhost:8000/?demo=1
```

## 已验证的演示能力

- 摄像头或上传视频实时姿态识别
- 自动/侧面/正面视角切换
- 深蹲阶段识别与次数统计
- 膝角、髋角、躯干倾角、膝前移或膝内扣实时展示
- 实时文字反馈、画质提示和每组动作质量评分
- 无摄像头时可直接运行演示模式

## 说明

- 视频画面和骨骼关键点只在浏览器本地处理，不上传服务器。
- 姿态模型使用 MediaPipe Pose Lite，模型文件已放在 `models/` 目录。
- 这是课堂原型，不是医学或正式运动指导工具。

## Cloudflare Pages 部署

- Build command: `npm run build`
- Build output directory: `dist`
- Framework preset: Static/None
