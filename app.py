import pennylane as qml
import numpy as np
import streamlit as st
from qutip import Bloch
import time


def calculate_arc_length(start, end):
    """
    半径1のBloch球上の2点の間のアーク長を計算する関数

    :param start: 3次元ベクトルとしての開始点 [x, y, z]
    :param end: 3次元ベクトルとしての終了点 [x, y, z]
    :return: アーク長
    """
    # 入力をnumpy配列に変換
    start = np.array(start)
    end = np.array(end)

    # 角度の計算
    cos_theta = np.dot(start, end) / (np.linalg.norm(start) * np.linalg.norm(end))
    theta = np.arccos(cos_theta)

    # アーク長の計算（半径1のBloch球の場合、アーク長は角度そのもの）
    arc_length = theta

    return arc_length


language = st.sidebar.radio("Language", ["English", "Japanese"], index=1)

is_game_mode = st.sidebar.checkbox(
    {"English": "Game mode", "Japanese": "🐔チキンレースに参加！"}[language]
)


if is_game_mode:
    seed = int(
        st.sidebar.number_input(
            {"English": "Random number", "Japanese": "乱数"}[language],
            min_value=-1,
            max_value=1000,
            value=0,
            step=1,
        )
    )

    # 時間制限の設定
    time_limit_in_seconds = 10

    # プログレスバーを表示
    time_progress_label = st.sidebar.empty()
    time_progress_bar = st.sidebar.progress(0)

    if st.sidebar.button(
        {"English": "Start!", "Japanese": "スタート"}[language], key="start"
    ):
        # スタートボタンを配置
        st.session_state.start_time = time.time()

    if "start_time" in st.session_state:
        # 経過時間の計算
        current_time = time.time()
        elapsed_time = current_time - st.session_state.start_time
        time_progress_label.write(
            {
                "English": f"Elapsed Time: {elapsed_time:.2f} sec",
                "Japanese": f"経過時間: {elapsed_time:.2f} （秒）",
            }[language]
        )

        # プログレスバーの進行状況を更新
        progress_ratio = elapsed_time / time_limit_in_seconds
        if progress_ratio > 1.0:
            st.warning("Time is up!")
            progress_ratio = 1.0
        time_progress_bar.progress(progress_ratio)

    if st.sidebar.button(
        {"English": "Stop!", "Japanese": "ストップ"}[language], key="stop"
    ):
        current_time = time.time()
        elapsed_time = current_time - st.session_state.start_time
        time_progress_label.write(
            {
                "English": f"Elapsed Time: {elapsed_time:.2f} sec",
                "Japanese": f"経過時間: {elapsed_time:.2f} （秒）",
            }[language]
        )

        # プログレスバーの進行状況を更新
        progress_ratio = elapsed_time / time_limit_in_seconds
        if progress_ratio < 1.0:
            time_progress_bar.progress(progress_ratio)
        else:
            time_progress_bar.progress(1.0)


# デバイスを設定
dev = qml.device("default.qubit", wires=1)

# 現在のゲートのリストを保存するためのセッション状態を初期化
if "gates_sequence" not in st.session_state:
    st.session_state.gates_sequence = []

# デフォルトの視点設定
default_azimuth = 295
default_elevation = 25

# 視点設定を保存するためのセッション状態を初期化
if "azimuth" not in st.session_state:
    st.session_state.azimuth = default_azimuth
if "elevation" not in st.session_state:
    st.session_state.elevation = default_elevation

# 初期状態のパラメータを設定するためのスライダー
if is_game_mode:
    # ランダムに初期状態を設定
    np.random.seed(seed)
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2 * np.pi)

    # ランダムなゴール状態を設定
    goal_theta = np.random.uniform(0, np.pi)
    goal_phi = np.random.uniform(0, 2 * np.pi)
else:
    st.sidebar.title(
        {"English": "Initial State Parameters", "Japanese": "初期状態のパラメータ"}[
            language
        ]
    )
    theta = st.sidebar.slider("Theta (θ)", 0.0, np.pi, 0.0)
    phi = st.sidebar.slider("Phi (φ)", 0.0, 2 * np.pi, 0.0)

# ゲートを追加するためのボタン
st.sidebar.title({"English": "Add Gate", "Japanese": "ゲートを追加"}[language])
if st.sidebar.button({"English": "X Gate", "Japanese": "X ゲート"}[language]):
    st.session_state.gates_sequence.append("X")
if st.sidebar.button({"English": "Y Gate", "Japanese": "Y ゲート"}[language]):
    st.session_state.gates_sequence.append("Y")
if st.sidebar.button({"English": "Z Gate", "Japanese": "Z ゲート"}[language]):
    st.session_state.gates_sequence.append("Z")
if st.sidebar.button({"English": "H Gate", "Japanese": "H ゲート"}[language]):
    st.session_state.gates_sequence.append("H")

# ゲートのシーケンスをリセットするためのボタン
if st.sidebar.button(
    {"English": "Reset Gates", "Japanese": "ゲートをリセット"}[language]
):
    st.session_state.gates_sequence = []

# 現在のゲートシーケンスを表示
st.sidebar.write(
    {"English": "Current Gate Sequence:", "Japanese": "ゲートの適用順序:"}[language]
)
st.sidebar.write(st.session_state.gates_sequence)

# デフォルトに戻すボタン
st.sidebar.title({"English": "Viewpoint Settings", "Japanese": "視点の設定"}[language])

if st.sidebar.button(
    {"English": "Reset Viewpoint to Default", "Japanese": "視点を初期設定に戻す"}[
        language
    ]
):
    st.session_state.azimuth = default_azimuth
    st.session_state.elevation = default_elevation

# Streamlitで視点を設定するスライダーを追加
azimuth = st.sidebar.slider(
    {"English": "Azimuth", "Japanese": "方位角"}[language], 0, 360, key="azimuth"
)
elevation = st.sidebar.slider(
    {"English": "Elevation", "Japanese": "仰角"}[language], -90, 90, key="elevation"
)


# ブロッホベクトルを計算する関数
def state_to_bloch_vector(state):
    rho = np.outer(state, np.conj(state))
    sx = np.array([[0, 1], [1, 0]])  # Pauli-X
    sy = np.array([[0, -1j], [1j, 0]])  # Pauli-Y
    sz = np.array([[1, 0], [0, -1]])  # Pauli-Z
    bx = np.trace(np.dot(rho, sx)).real
    by = np.trace(np.dot(rho, sy)).real
    bz = np.trace(np.dot(rho, sz)).real
    return np.array([bx, by, bz])


# 初期状態を設定する関数
def initialize_state(theta, phi):
    return np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])


# ゲートを適用する関数
def apply_gate(state, gate, steps=100):
    if gate == "X":
        axis = np.array([1, 0, 0])
    elif gate == "Y":
        axis = np.array([0, 1, 0])
    elif gate == "Z":
        axis = np.array([0, 0, 1])
    elif gate == "H":
        axis = np.array([1, 0, 1]) / np.sqrt(2)
    else:
        raise ValueError(f"Unknown gate: {gate}")

    angle = np.pi
    new_states = []
    for t in np.linspace(0, angle, steps):
        rotation_matrix = np.cos(t / 2) * np.eye(2) - 1j * np.sin(t / 2) * (
            axis[0] * np.array([[0, 1], [1, 0]])
            + axis[1] * np.array([[0, -1j], [1j, 0]])
            + axis[2] * np.array([[1, 0], [0, -1]])
        )
        new_state = np.dot(rotation_matrix, state)
        new_states.append(new_state)
    return new_states


# 初期状態を設定
state = initialize_state(theta, phi)
states = [state]

if is_game_mode:
    # ゴール状態を設定
    goal_state = initialize_state(goal_theta, goal_phi)

    # ゴール状態をベクトルとして追加
    goal_bloch_vector = state_to_bloch_vector(goal_state)

# 各ゲート適用後の状態を保存するリスト
final_states = [state]

# ゲートシーケンスを順に適用
for gate in st.session_state.gates_sequence:
    new_states = apply_gate(states[-1], gate)
    states.extend(new_states)
    final_states.append(new_states[-1])

# 各状態のブラッホベクトルを計算
bloch_vectors = [state_to_bloch_vector(state) for state in states]
final_bloch_vectors = [state_to_bloch_vector(state) for state in final_states]

# Streamlitでブロッホ球を描画
st.title(
    {
        "English": "Bloch Sphere Visualization: Apply Gates",
        "Japanese": "🐓量子チキンレースにようこそ！🐓",
    }[language]
)

b = Bloch()
b.view = [azimuth, elevation]

# 軌跡を追加
if bloch_vectors:
    trajectory = np.array(bloch_vectors).T  # 軌跡を2D配列として追加
    b.add_points(trajectory, meth="l")  # 線形補間を使用

# 初期状態と終端状態を点として追加
b.add_points(final_bloch_vectors[0])
b.add_points(final_bloch_vectors[-1])
if is_game_mode:
    b.add_annotation(final_bloch_vectors[0], "start")
    b.add_annotation(
        final_bloch_vectors[-1],
        r"$|\psi\rangle$",
    )

# 各状態をベクトルとして追加
for bloch_vector in final_bloch_vectors[0:]:
    b.add_vectors(bloch_vector)

if is_game_mode:
    # ゴール状態をベクトルとして描画
    b.add_vectors(goal_bloch_vector, colors=["black"])
    b.add_annotation(goal_bloch_vector, "goal")

    b.add_arc(
        final_bloch_vectors[-1], goal_bloch_vector, fmt="r--"
    )  # 終端ベクトルからゴールベクトルへの赤い円弧を追加

    remaining_distance = calculate_arc_length(
        final_bloch_vectors[-1], goal_bloch_vector
    )

    # 終端ベクトルからゴールベクトルまでの残りの距離を表示
    st.info(
        {
            "English": f"Remaining Distance: {remaining_distance:.2f} / {np.pi:.2f}",
            "Japanese": f"ゴールまでの距離:  {remaining_distance:.2f} / {np.pi:.2f}",
        }[language]
    )
    st.progress(remaining_distance / (np.pi))


# ベクトルの先の移動距離を計算
def calculate_total_distance(vectors):
    total_distance = 0.0
    for i in range(1, len(vectors)):
        total_distance += np.linalg.norm(vectors[i] - vectors[i - 1])
    return total_distance


if bloch_vectors:
    total_distance = calculate_total_distance(bloch_vectors)

    # 総移動距離を表示
    if is_game_mode:
        distance_threshold = st.sidebar.number_input(
            {"English": "Distance Threshold", "Japanese": "移動距離の閾値"}[language],
            value=4 * 3.14,
            step=0.01,
        )

        st.info(
            {
                "English": f"Total Distance Traveled: {total_distance:.2f} / {distance_threshold:.2f}",
                "Japanese": f"総移動距離: {total_distance:.2f} / {distance_threshold:.2f}",
            }[language],
        )

    else:
        st.info(
            {
                "English": f"Total Distance Traveled: {total_distance:.2f}",
                "Japanese": f"総移動距離: {total_distance:.2f}",
            }[language],
        )

    if is_game_mode:
        distance_ratio = total_distance / distance_threshold

        if total_distance > distance_threshold:
            st.progress(1.0)
            st.image("static/images/science_hakase_shippai.png")
            st.warning(
                {
                    "English": "The total distance traveled is large. Consider reducing the number of steps.",
                    "Japanese": "ドボン！総移動距離が大きすぎます！ゲートの数を減らしてみましょう。",
                }[language]
            )
            st.balloons()
        else:
            st.progress(distance_ratio)

# ブロッホ球を表示
b.render()
st.pyplot(b.fig)
