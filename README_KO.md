# ChatMind

**채팅 메시지 시맨틱 검색** - TurboQuant 벡터 압축 (ICLR 2026) 기반

Discord, 카카오톡 등의 채팅 내보내기 파일에서 **의미로 검색**하세요.

> *"친구가 추천해준 맛집이 뭐였지?"*
> → 정확한 단어를 기억 못해도 ChatMind가 찾아줍니다.

## 왜 ChatMind인가?

| 기능 | Discord 검색 | Ctrl+F | **ChatMind** |
|------|-------------|--------|-------------|
| 의미 검색 | X | X | **O** |
| 동의어 인식 | X | X | **O** |
| 유사도 점수 | X | X | **O** (0.0~1.0) |
| 오프라인 | X | O | **O** |
| 다국어 검색 | X | X | **O** |
| 발신자/날짜 필터 | 일부 | X | **O** |

## 설치

```bash
pip install chatmind
```

## 사용법

### 1. Discord 채팅 내보내기

[DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter)로 JSON 내보내기

### 2. 인덱싱 & 검색

```bash
# 인덱싱
chatmind index discord_export.json

# 의미 검색
chatmind search "맛집 추천"
chatmind search "여행 계획" --from Alex
chatmind search "숙제 도움" --after 2024-01-01
```

### 출력 예시

```
Search: 맛집 추천
Found 5 results

  #1  [HIGH]  0.89  2024-01-15 10:02
      Park: I found an amazing sushi restaurant near Gangnam station
      [general]

  #2  [HIGH]  0.84  2024-01-15 14:32
      Park: Try the pasta place on 5th street, their carbonara is amazing
      [general]
```

## 지원 플랫폼

| 플랫폼 | 형식 | 내보내기 방법 |
|--------|------|--------------|
| **Discord** | `.json`, `.csv` | [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter) |
| **카카오톡** | `.txt` | 채팅방 > 메뉴 > 대화 내보내기 |

## 명령어

```bash
chatmind index <file>              # 채팅 인덱싱
chatmind search "검색어"            # 시맨틱 검색
chatmind search "검색어" --from Kim  # 발신자 필터
chatmind search "검색어" --room gen  # 채널 필터
chatmind stats                     # 통계
chatmind rooms                     # 채널 목록
chatmind people                    # 참여자 목록
```

## 관련 프로젝트

- [CommitMind](https://github.com/wjddusrb03/commitmind) - Git 커밋 시맨틱 검색
- [langchain-turboquant](https://github.com/wjddusrb03/langchain-turboquant) - LangChain용 TurboQuant

## 피드백

버그나 아이디어가 있으면 [Issue](https://github.com/wjddusrb03/chatmind/issues)를 열어주세요!
한국어/영어 모두 환영합니다.

## 라이선스

MIT License
