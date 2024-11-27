# 本shell脚本由伟大的开源程序员曹梦轩贡献和维护
# 本脚本每 30 秒监测一次 GPU 温度，并根据温度阈值调整功率上限以控制频率。
# 当 GPU 温度超过指定上限温度时，将功率上限设置为降频功率值，以降低频率。
# 当 GPU 温度降至恢复温度以下时，恢复功率上限为正常功率值。
# 上限温度、恢复温度、降频功率和正常功率均作为参数输入，方便灵活调整。
# 所有的操作和状态变更会记录在日志文件中，以便后续查看 GPU 状态和调整历史。
# ps aux | grep 'monitor_gpu' | grep -v 'grep'   使用这个指令可以找到后台运行的进程，友情提示。

# 输出启动信息到控制台
echo "$(date): GPU 温控监测脚本启动..." | tee /dev/tty

# 参数设置
UPPER_TEMP=85                # 触发降频的温度上限
LOWER_TEMP=75                # 恢复正常频率的温度下限
LOW_POWER=100                # 降频时的功率上限（单位：瓦）
HIGH_POWER=400               # 恢复正常频率时的功率上限（单位：瓦）
LOG_FILE="/var/log/gpu_monitor.log"  # 日志文件路径

# 确保日志文件可写
touch "$LOG_FILE" && chmod 644 "$LOG_FILE"

# 初始化 count 为 0
count=0

# 后台运行脚本的循环
monitor_gpu_temp() {
    # 定义低功率计数器
    declare -A low_power_counter
    while true; do
        # 获取每个GPU的温度，逐行处理
        nvidia-smi --query-gpu=index,temperature.gpu --format=csv,noheader,nounits | while read -r index temp; do
            # 去除温度字符串中的空格和逗号，确保正确解析
            index=$(echo $index | tr -d '[:space:],')
            temp=$(echo $temp | tr -d '[:space:],')
            
            # 确保 low_power_counter[$index] 已初始化
            if [[ -z "${low_power_counter[$index]}" ]]; then
                low_power_counter[$index]=0
            fi

            # 输出当前的温度，调试脚本才打开用
            #echo "$(date): 当前温度检查 - GPU $index: ${temp}°C" | tee -a "$LOG_FILE"

            # 每 10 轮（即 5 分钟）检查一次
            if (( count % 10 == 0 )); then
                echo "$(date): GPU $index 当前温度为 ${temp}°C" | tee -a "$LOG_FILE"
            fi

            # 判断温度并进行操作
            if (( temp > UPPER_TEMP )); then
                # 温度超过上限，降频处理
                echo "$(date): GPU $index 温度为 ${temp}°C，开始降频至 ${LOW_POWER}W..." | tee -a "$LOG_FILE"
                echo "警告：GPU $index 温度过高！当前温度为 ${temp}°C，降频至 ${LOW_POWER}W。"
                nvidia-smi -i $index -pl $LOW_POWER
                low_power_counter[$index]=1
            elif (( temp < LOWER_TEMP )) && (( low_power_counter[$index] == 1 )); then
                # 只有当温度低于下限且当前处于低功率状态时才恢复功率
                echo "$(date): GPU $index 温度已降至 ${temp}°C，恢复功率上限至 ${HIGH_POWER}W..." | tee -a "$LOG_FILE"
                echo "信息：GPU $index 温度恢复正常，当前温度为 ${temp}°C，恢复功率上限至 ${HIGH_POWER}W。"
                nvidia-smi -i $index -pl $HIGH_POWER
                low_power_counter[$index]=0
            else
                # 温度在正常范围内，维持当前功率
                if (( count % 20 == 0 )); then
                    if [ "${low_power_counter[$index]}" -eq 1 ]; then
                        # 如果处于低功率状态
                        nvidia-smi -i $index -pl $LOW_POWER
                        echo "$(date): GPU $index 温度为 ${temp}°C，当前功率上限为 ${LOW_POWER}W，维持当前功率设置。" | tee -a "$LOG_FILE"
                    else
                        # 如果处于正常功率状态
                        nvidia-smi -i $index -pl $HIGH_POWER
                        echo "$(date): GPU $index 温度为 ${temp}°C，当前功率上限为 ${HIGH_POWER}W，维持当前功率设置。" | tee -a "$LOG_FILE"
                    fi
                fi
            fi
        done

        # 每30秒进行一次监测
        sleep 30
        count=$((count + 1))
    done
}

# 启动温控监测
monitor_gpu_temp &

