1. 새롭게 작성되거나 내부 코드가 변경된 파일만 업로드 하였습니다.
`firehose_pz_env.py`
위치: firehose/cell2fire

`gym_env.py`
위치: firehose/cell2fire

`rewards_multi.py`
위치: firehose/cell2fire/firehose

`train_firehose_rllib.py`
위치: firehose/cell2fire

`test_firehose_pz.py`
위치: firehose/cell2fire

`rl_experiment_multiagent.py`
위치: firehose/cell2fire

2. 새롭게 설치한 라이브러리나 모듈 목록입니다.
```python
pip install gymnasium\
pip install pettingzoo\
pip install "ray[rllib]"\
pip install torch\
```
3. Iteration 100회 정도 돌려본 결과도 업로드 해두었습니다.
`ray_tune_env_runners_agent_episode_return ~` 으로 시작하는 세 파일은 각각 헬리콥터, 드론, 지상인력의 return 그래프입니다.\
`ray_tun_env_runners_agent_episode_return~` 으로 시작하는 하나의 파일은 total return 그래프입니다.\
우선 제가 대충 reward를 짜서 돌려본거라 return이 세 agent 모두에게서 증가되진 않습니다.\
혹시나 terminal에 찍힌 로그들도 도움이 될까하여 `multiagent_output.txt`도 업로드해두었습니다.\
