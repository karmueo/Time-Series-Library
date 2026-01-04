def fun_tar_info(data, start):


    status = struct.unpack("=1B", data[start:start + 1])[0]
    status = status % 16
    trk_stat = status
    # 批号
    tar_id = struct.unpack("=1H", data[start + 2:start + 4])[0]
    # 流水号
    tar_liu = struct.unpack("=1I", data[start + 6:start + 10])[0]
    # # 航迹历史
    trk_cn = struct.unpack("=1H", data[start + 10:start + 12])[0]

    # ICAO = 0

    # 时间
    tim = struct.unpack("=1I", data[start + 22:start + 26])[0]
    tim = tim * 25e-6

    # 滤波后RAE
    r, a, e = struct.unpack("=2I1i", data[start + 26:start + 38])
    # r,a,e转换为米、度、度
    r = r * 1e-1
    a = a * 1e-5
    e = e * 1e-5


    # 点迹RAE
    pr, pa, pe = struct.unpack("=2I1i", data[start + 38:start + 50])
    # r,a,e转换为米、度、度
    pr = pr * 1e-1
    pa = pa * 1e-5
    pe = pe * 1e-5


    # 直角坐标
    x1 = 0
    y1 = 0

    # 高度
    h1 = (pr * np.sin(pe * pi / 180) + pr * pr / 17000000)

    # 目标全速度
    vel = struct.unpack("=1I", data[start + 58:start + 62])[0]
    vel = vel * 0.1 #m/s

    # 目标空间加速度
    at = 0  # m/s

    # #航向角
    # course_angle = 0  #度

    # # 目标幅度，信噪比、RCS
    snr, rcs = struct.unpack("=1H1h", data[start + 80:start + 84])
    snr = snr * 0.01 #dB
    rcs = rcs * 0.01#dB

    # 目标类型
    t1 = struct.unpack("=1H", data[start + 84:start + 86])[0]
    tarBigClass = t1 % 256
    tarSmaClass = int(t1 / 256)

    # 目标JEM特征
    feat1 = struct.unpack("=1f", data[start + 152:start + 156])[0]
    feat2 = struct.unpack("=1f", data[start + 156:start + 160])[0]

    # feat3 = 0
    # feat4 = 0
    # feat5 = 0

    n_data = {
        '目标状态':[status],
        '航迹状态':[trk_stat],
        '航迹历史': [trk_cn],
        'track_id': [tar_id],
        '目标流水号': [tar_liu],
        # '民航号':[ICAO],
        'timestamp': [tim],
        # '目标幅度': [tarA],
        'RCS': [rcs],
        'SNR': [snr],
        'R': [r],
        'A': [a],
        'E': [e],
        '点迹距离': [pr],
        '点迹方位': [pa],
        '点迹俯仰': [pe],
        '高度':[h1],
        '全速度': [vel],
        # '加速度':[at],
        # '航向角':[course_angle],
        '航线角': [0.0],
        '航线差':[0.0],
        'Feature1': [feat1],
        'Feature5': [feat2],
        # '杂波背景': [feat3],
        # '特征4': [feat4],     #对称峰值
        # '特征5': [feat5],     #自相关，峰值差
        # '特征5大于门限次数': [feat5_num],  # 自相关，峰值差
        '目标大类': [tarBigClass],
        '目标小类': [tarSmaClass],
        'x1':[x1],
        'y1':[y1]

    }

    df = pd.DataFrame(n_data)
    return df




# -------------------------------
# 📡 接收线程：只负责接收原始数据
# -------------------------------
def receiver_thread():
    """📥 仅做一件事：接收 JSON 点迹，放入 raw_point_queue"""
    print(f"🔁 启动接收线程 → {MULTICAST_RCV_ADDR}:{MULTICAST_RCV_PORT}")

    sock = None
    try:
        if SRC_FLG == 0:  # 0:网络 1：文件
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('', MULTICAST_RCV_PORT))

            mreq = struct.pack("4s4s", socket.inet_aton(MULTICAST_RCV_ADDR), socket.inet_aton('0.0.0.0'))
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            sock.settimeout(1)

        ############### 文件回放
        else:  #1：文件
            trk_flg1 = 0x1010  # 航迹起始标志   0x5100：110  0x1010 101
            trk_flg2 = 0x55AA  # 航迹结束标志
            tar_str = 26  # 目标位置偏移起始
            tar_len = 160  # 目标信息长度 216：110  160:101

            with open(TRK_FILE, 'rb') as f:
                trk_dat = f.read()
            trk_flg21 = trk_flg2 & 0xFF
            trk_flg22 = (trk_flg2 >> 8) & 0xFF

            trk_flg11 = trk_flg1 & 0xFF
            trk_flg12 = (trk_flg1 >> 8) & 0xFF

            cur_list = []
            last_cur = 0
            for n in range(len(trk_dat) - 4):
                if trk_dat[n] == trk_flg21 and trk_dat[n + 1] == trk_flg22 and trk_dat[n + 2] == trk_flg11 and trk_dat[n + 3] == trk_flg12:
                    cur_list.append(n + 2)

            fn = 0
            fn_len = len(cur_list)


        buffer = ""
        while running:
            if SRC_FLG == 0: #网络
                try:
                    data, addr = sock.recvfrom(1024)
                    buffer += data.decode('utf-8', errors='ignore')
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            packet = json.loads(line)
                            if packet.get("type") == "point":
                                raw_point_queue.put(packet)  # 👉 投递到处理队列
                        except Exception as e:
                            print(f"❌ 解析失败: {e} -> {line[:100]}")
                except socket.timeout:
                    continue
            else:    #文件
                if fn > fn_len - 2:
                    print('--fnish read ' + TRK_FILE)

                    time.sleep(1)
                    print("程序即将退出")
                    sys.exit(0)  # 0 表示正常退出，非 0 表示异常退出
                    break

                data = trk_dat[cur_list[fn]:cur_list[fn + 1]]

                if fn % 10 == 0:
                    time.sleep(1)  # 休眠1ms

                if fn % 1000 == 0:
                    print('--read--' + str(fn_len) + '-' + str(fn))

                fn = fn + 1

            ########## 解析
            dat_len = len(data)
            tar_num = 0
            df1 = pd.DataFrame()
            num = int(struct.unpack("=1H", data[24:26])[0])  # 本包内目标数
            for i in range(num):
                # 26是报文头加点迹数 160 是一段航迹长度 38是点迹距离起始
                start = tar_str + i * tar_len
                if (start + tar_len) > dat_len:
                    print('--1-datLen--')
                    break

                # # 目标状态
                # t1 = struct.unpack("=1B", data[start + 25:start + 26])[0]
                # status = t1 / 16  # 0-丢失   1-跟踪   2-记忆 3-消批
                # if status==2:
                #     continue

                df = fun_tar_info(data, start)  # 目标数据解析
                tar_num = tar_num + 1

                # tar_id = df['目标批号'].values[0]
                # tar_liu = df['目标流水号'].values[0]
                # status = df['目标状态'].values[0]
                # tar_cn = df['航迹历史'].values[0]
                row_dict = df.iloc[0].to_dict()  # 转成字典再放进去
                raw_point_queue.put(row_dict)



    except Exception as e:
        print(f"⚠️ 接收线程异常: {e}")
    finally:
        if sock:
            sock.close()

