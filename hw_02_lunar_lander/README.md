# RL Homework

## How to submit
Задачи отправляются боту в Telegram: `@RL_hw_bot`. В качестве решения задачи принимается zip-архив с кодом. Важно: содержимое папки с заданием должно быть в корне, иначе бот не найдет ваше решение.

## Task
В данном задании необходимо научить агента побеждать в игре LunarLander при помощи метода DQN или одной из его модификаций.

К заданию нужно приложить код обучения агента (не забудьте зафиксировать seed!), готовый (уже обученный) агент должен быть описан в классе Agent в файле `agent.py`.

## Оценка:
От `1` до `10`, баллы начисляются за полученные агентом очки в среднем за `50` эпизодов. Максимальный балл соответствует `200` очкам и выше, минимальный - `-100`.