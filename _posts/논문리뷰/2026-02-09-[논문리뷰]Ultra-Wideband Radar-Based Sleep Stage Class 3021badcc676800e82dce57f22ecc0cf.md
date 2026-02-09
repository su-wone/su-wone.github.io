# [논문리뷰]Ultra-Wideband Radar-Based Sleep Stage
Classification in Smartphone Using
an End-to-End Deep Learning

## ABSTRACT

---

최근 불면증이나 수면무호흡증 같은 **수면장애**로 고통받는 사람이 증가하면서, 일반 소비자용 기기를 이용한 수면 모니터링·관리 기술에 대한 관심이 커지고 있다. 수면의 질은 하이프노그램(hypnogram)에 기반한 수면 구조와 밀접하게 연관되어 있기 때문에, 밤 동안의 **수면 단계(sleep stage)를 정확히 분류**하는 것이 필수적이다.

이 논문에서는 **초광대역(UWB) 레이더가 장착된 스마트폰**을 이용한 수면 단계 분류 방법을 제안한다. 대부분의 사람이 잠잘 때 스마트폰을 침대 옆 탁자에 두는 생활 패턴에 맞춰, 스마트폰을 단지 침대 옆 탁자에 두기만 하면 누구나 쉽게 사용할 수 있는 수면 모니터링 시스템을 목표로 했다.

연구진은 커스터마이즈된 삼성 갤럭시 스마트폰(내부에 UWB 레이더 칩 탑재)을 사용해, 침대 옆 탁자에 스마트폰을 놓은 상태에서 **509박의 UWB 레이더 데이터와 야간 PSG(실험실 다채널 수면다원검사) 데이터를 동시에 수집**했다. 여기에는 정상인뿐 아니라 **수면무호흡증 환자도 포함**되었다.

레이더 데이터가 부족하다는 한계를 극복하기 위해, **1D CNN + Transformer**로 구성된 **엔드투엔드 딥러닝 모델**을 제안하고, 대규모 공개 PSG 데이터베이스에서 얻은 호흡 신호(스펙트로그램)와 UWB 레이더 데이터를 함께 활용하는 **도메인 적응(domain adaptation) 기법**을 적용했다. 이를 통해 레이더 데이터만으로 학습했을 때보다 성능을 향상시켰다.

5-fold 교차검증을 통해, 예측된 네 가지 수면 단계(각성, REM, 가벼운 수면, 깊은 수면)와 전문가 PSG 라벨을 에포크 단위(30초)로 비교한 결과, **정확도 0.76, Cohen’s kappa 0.64**를 얻었다. 이는 단순히 스마트폰을 침대 옆 탁자에 두는 것만으로도 **상당히 신뢰할 수 있는 수준의 수면 단계 모니터링이 가능함**을 보여주며, 실제 사용성 측면에서 매우 유용한 시스템임을 입증한다.

## INTRODUCTION

---

- 수면은 인간의 건강과 웰빙에 매우 중요한 역할을 한다.
- 여러 연구에 따르면, **수면의 질이 낮으면 분자·면역·신경계 수준에서 다양한 변화가 발생하고, 이는 여러 질환의 발병에 영향을 줄 수 있다.**
- 최근에는 불면증이나 수면무호흡증 같은 수면장애를 겪는 인구가 증가하면서, **일반 소비자용 기기를 활용한 수면 모니터링 및 관리**에 대한 관심도 함께 커지고 있다.

수면의 질은 하이프노그램(hypnogram)에 의해 표현되는 **수면 구조(sleep architecture)**와 밀접하게 관련되어 있다. 따라서 밤 전체에 걸친 **수면 단계(수면 단계별 분포)를 정확히 분류하는 것**이 신뢰할 수 있는 수면 모니터링을 위해 중요하다.

 현재 수면 단계 판정의 골드 스탠다드는 **수면다원검사(PSG)**이며, 이는 뇌파(EEG), 심전도(ECG), 안구운동(EOG), 근전도(EMG) 등 다양한 생체신호를 여러 부위에 부착된 센서로 측정한다.

<단점>

그러나 PSG는 전극을 많이 붙여야 해서 **착용감이 불편하고 수면 자체를 방해**할 수 있으며, 숙련된 기술자가 전문 수면센터에서만 수행할 수 있다. 비용·시간·노동력이 많이 들기 때문에, 가정에서 장기간 자동으로 수면을 모니터링하기에는 적합하지 않다.

<단점, 한계 극복>

이 한계를 줄이기 위해, 몇 개의 센서만 사용한 **자동 수면 단계 분류 연구**가 많이 진행되었다. 특히, 수면 단계는 뇌파 패턴과 밀접하므로 **단일 채널 EEG 기반 딥러닝/머신러닝 모델**이 활발히 연구되었다. 동시에, 손목밴드·스마트워치처럼 가속도계와 심박 센서를 내장한 웨어러블 디바이스를 이용한 수면 모니터링도 활발히 개발되어 왔다.

<여전한 단점, 한계>

하지만 **몸에 센서를 붙이거나 웨어러블을 밤새 착용하는 것 자체가 여전히 불편**할 수 있다. 이런 이유로, 최근에는 신체에 직접 닿지 않고 사용하는 **비접촉(non-contact) 수면 모니터링 장치**, 이른바 **Nearables**가 등장했다. 예를 들어, 일부 기기는 매트리스 아래/위에 설치되어 ballistocardiogram, 호흡, 움직임 등을 이용하고(EarlySense, Emfit, Withings Aura, Beddit, Eight, RestOn 등), 또 다른 기기들은 침대 주변에 설치되어 소리나 레이더를 사용한다(SleepScore Max, Google Nest Hub 2, Amazon Halo Rise, ResMed S+ 등). 그러나 이런 Nearable과 웨어러블 기기들은 **별도의 제품 구매·설치가 필요해 비용과 접근성 측면에서 제약**이 있다.

이에 따라, 추가 비용 없이 거의 모든 사람이 이미 가지고 있는 **스마트폰을 수면 모니터링에 활용하려는 시도**가 이루어지고 있다. 

<스마트폰 이용 도입 근거>

한 조사에 따르면, 미국인의 약 4분의 3이 **스마트폰을 침대 근처에 둔 채로 잠을 잔다**고 하며, 이 때문에 스마트폰은 다른 상용기기보다 수면 모니터링에 더 적합한 플랫폼이 될 수 있다. 기존에는 스마트폰의 **마이크를 이용해 수면 상태를 추정하는 연구**들이 있었지만, 성능이 충분히 만족스럽지 못했고, 사운드 기반 접근은 **프라이버시 문제**도 내포하고 있다.

<논문의 해결방법>

이 논문에서는 이러한 문제를 해결하기 위해, 스마트폰에 **초광대역(UWB) 레이더를 탑재하고, 단지 침대 옆 탁자에 스마트폰을 올려두는 것만으로 수면을 모니터링하는 방법**을 제안한다. UWB 레이더는 **호흡·심장박동·몸의 미세 움직임에 따른 도플러 변화**를 포착할 수 있지만, **스마트폰이 침대 탁자에 놓인 현실적인 배치에서는 레이더와 몸의 상대 위치에 따라 신호대잡음비(SNR)가 낮아지고, 호흡률과 같은 전통적인 특징을 안정적으로 추출하기 어렵다**는 문제가 있다.

<해결방법>

이를 해결하기 위해 저자들은 **수작업 특징 추출을 생략하고, 1D CNN과 Transformer로 구성된 엔드투엔드 딥러닝 모델**을 사용해 도플러 맵(Doppler map)에서 직접 특징을 학습하도록 설계했다. 또한, **PSG 공개 데이터베이스(National Sleep Research Resource)의 대규모 호흡 신호 스펙트로그램**과 **스마트폰 UWB 레이더 도플러 맵**을 함께 활용하는 **도메인 적응(domain adaptation)** 기법을 도입해, 레이더 데이터 부족 문제를 극복하고 성능을 향상시켰다.

<연구 진행방향>

연구에서는 정상인과 수면무호흡 환자를 포함한 **다양한 인구 집단**으로부터 레이더·PSG 데이터를 수집하고, 제안한 알고리즘을 검증했다. 저자들은 이 시스템이 **추가 하드웨어 없이 스마트폰만으로 수면 단계 모니터링을 상당한 정확도로 수행할 수 있어, 일반 대중에게 높은 사용성과 접근성을 제공할 수 있다**고 결론 내린다.

## METHODS

---

### A. PARTICPANT

---

- 총 **290명**의 참가자가 내부 검증용 데이터셋에 포함되었음.
    - 한국 KOSLEEP 수면클리닉: 110명
    - 브라질 São Paulo의 Instituto do Sono: 180명
- 모든 참가자는 서면 동의서를 제출했고,
    - 한국 쪽은 보건복지부 지정 공용기관생명윤리위원회(P01-202109-11-006),
    - 브라질 쪽은 상파울루 연방대학 윤리위원회(0081/2021)의 승인을 받음.
- 추가로, **삼성서울병원(SMC)**에서 **56명**을 별도 모집하여 **외부 검증(external validation)**에 사용.
    - 이들 역시 서면 동의를 했고, 보건복지부 지정 공용 IRB(2022-08-029-001)의 승인을 받음.
- **모든 기관의 포함 기준**:
    - 건강한 참가자 또는 수면무호흡(apnea) 환자
    - 심장 질환 또는 관상동맥 질환 진단을 받은 사람은 제외
- 논문 Table 2에는 각 기관별 인구통계 및 수면 단계 분포가 정리되어 있음.

![스크린샷 2026-02-09 오후 3.11.08.png](%5B%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0%5DUltra-Wideband%20Radar-Based%20Sleep%20Stage%20Class/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-02-09_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.11.08.png)

### B.PROTOCOL

---

수면 기록 횟수

- KOSLEEP 참가자 110명은 **서로 다른 3박(연속 아님)** 동안 PSG + UWB 레이더가 동시에 기록되도록 설계.
- Instituto do Sono(180명)와 SMC(56명) 참가자들은 **각각 1박만** 기록.
- KOSLEEP에서 기록 3박 중 1박은 기록 오류로 제외되어 분석에서 빠짐.

스마트폰 + UWB 레이더 배치 조건

- 레이더가 탑재된 스마트폰은, 사람들이 실제로 잠잘 때 자주 하는 것처럼 **침대 옆 탁자에 스마트폰을 두는 상황**을 그대로 모사.
    - 스마트폰은 **침대 옆 탁자 위에, 화면이 아래를 향하도록 엎어놓고**,
    - **폰의 모서리(edge)가 참가자를 향하도록 배치**했다(Fig. 1(b)).
- 추가 학습 데이터를 확보하기 위해, 레이더 위치를 엄격히 제어한 **삼각대(tripod) 세팅**도 별도로 사용.
    - 이 경우 스마트폰은 삼각대에 장착되고, **폰의 뒷면(back side)이 피험자를 향하도록** 배치.
- 두 경우 모두, 스마트폰(레이더)와 피험자 간 거리는 **약 1 m**로 유지.
    
    ![스크린샷 2026-02-09 오후 3.14.46.png](%5B%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0%5DUltra-Wideband%20Radar-Based%20Sleep%20Stage%20Class/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-02-09_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.14.46.png)
    

학습·평가에 사용한 데이터 구분

- **두 가지 세팅(탁자 위, 삼각대)의 데이터 모두를 모델 학습에 사용**.
- 하지만, 논문에서 **성능을 보고할 때는 “탁자 위에 스마트폰을 둔 실제 사용 시나리오”에 해당하는 데이터만 테스트셋으로 사용**했다.
    - 연구의 목적이 “현실적인 스마트폰 사용 환경에서 수면 모니터링 성능을 보여주는 것”이기 때문.

### **C. POLYSOMNOGRAPHY (PSG)**

---

장비 및 세팅

- PSG는 **AASM(American Academy of Sleep Medicine) 가이드라인**에 따라 수행.
- 사용 장비:
    - KOSLEEP, SMC: Embla N7000 (Natus, US)
    - Instituto do Sono: Alice6 LDx (Philips, Netherlands)

기록된 생체신호

- EEG: A1, F3, F3-A2, O1, O2-A1 채널
- ECG
- EMG: 양측 앞정강이(tibialis leg left/right)
- EOG: left A1 (좌측 기준)
- 호흡 신호 (respiratory signals)
- 산소 포화도(SpO₂)

수면 단계 라벨링

- 수면 단계는 **30초(epoch) 단위**로, AASM 기준에 따른 전문가 수기 라벨링 수행.
- 5단계 라벨:
    - Wake
    - N1
    - N2
    - N3
    - REM (Rapid Eye Movement)
- 이후 분석을 위해 단계 수를 줄인 레이블도 사용:
    - 4-stage: Wake, REM, Light (N1+N2), Deep (N3)
    - 3-stage: Wake, REM, NREM (Light + Deep)

## **D. IR-UWB RADAR**

---

### 스마트폰 및 레이더 하드웨어

- 스마트폰: Galaxy S21+ (Samsung Electronics, Korea)에 **IR-UWB radar-on-chip**을 커스터마이즈하여 장착.
- 레이더 칩: NXP UWB radar-on-chip (NXP Semiconductors, Netherlands).
- 동작 특성:
    - 주파수 범위: 6.24–8.24 GHz
    - 대역폭: 500 MHz
- 레이더칩은 수신 신호를 디지털화하고, 스마트폰 메인 시스템으로 전송하여 저장.

### 안테나 구성 및 배치

- 침대 옆 탁자 위에 둔 스마트폰 세팅:
    - 스마트폰 측면(lateral side)에 금속 안테나 2개 부착
        - 1개: 송신(TX)
        - 1개: 수신(RX)
- 삼각대(tripod) 세팅:
    - 스마트폰 측면에 TX용 금속 안테나
    - 스마트폰 뒷면(back side)에 RX용 패치 안테나 부착 (금속 안테나보다 큰 면적)
    - 대상자 쪽을 정면으로 조준하도록 설치
- 안테나의 **빔 패턴은 거의 무지향성(omnidirectional)**으로 설계되어, 대상자와 안테나 방향이 완벽히 정렬되지 않아도 동작 가능하도록 함.
    - 하지만 **SNR 향상을 위해서는 가능한 한 대상자와 안테나가 잘 정렬되도록 배치하는 것이 권장**된다고 명시.

## 레이더 신호의 물리적 의미

- IR-UWB는 **좁은 펄스 신호를 발사**하고, 대상에서 반사되어 돌아오는 시간(time-of-flight)을 측정하여 **거리(range)를 추정**함.
- 대상이 움직이면 송수신 주파수 차이가 발생하는 **도플러 효과(Doppler effect)**가 나타남.
    - 수면 중 흉곽 움직임(호흡), 심박 등에 의해 발생하는 미세 움직임이 여기에 포함됨.
- 여러 움직임에 의해 생기는 미세한 도플러 변화를 **시간–도플러 주파수 도메인에서 표현한 것이 micro-Doppler spectrogram, 즉 Doppler map**.
    - 이 Doppler map에는 **호흡, 심박, 자세 변경 등 수면 중 생체신호 및 움직임 패턴**이 풍부하게 포함됨.

## Doppler map 생성 파이프라인

1. **원시 레이더 샘플링**
    - 샘플링 주파수: 200 Hz
    - range bin 개수: 56개
    - range bin 간격: 15 cm
2. **다운샘플링**
    - 수면 관련 생리 신호(호흡, 심박)의 주요 주파수는 **2.5 Hz 이하**로,
    - Doppler map의 도플러 주파수 범위를 –2.5 ~ 2.5 Hz로 맞추기 위해 **레이더 신호를 5 Hz로 다운샘플링**.
3. **클러터 제거 (moving average)**
    - 각 range bin에 대해, 12초 길이 윈도우를 사용한 **이동 평균**으로 클러터(clutter) 추정.
    - 원시 레이더 신호에서 이 클러터를 빼서, 대상자의 움직임에 의한 성분만 강조.
4. **주파수 도메인 변환**
    - 각 range bin 신호에 대해, **256-point sliding DFT**(윈도우 길이 12초) 적용.
    - 이를 통해 시간에 따른 도플러 주파수 스펙트럼을 구함.
5. **range bin 통합**
    - 모든 range bin에서 얻은 Doppler 스펙트럼을 합산하여,
    - 레이더의 시야 내 대상 전체 움직임을 대표하는 **합성(composite) Doppler map** 생성.
6. **PSG와 시간 동기화**
    - UWB Doppler map에서 추출한 호흡 신호와, PSG 호흡 신호 간의 **상호 상관(cross-correlation)을 최대화**하는 방식으로 타임 얼라인먼트.

### 모델 입력용 전처리 (시간·주파수 축 가공)

- 시간축:
    - 모델에 넣을 때는 30초(epoch) 기준으로 보려 하므로,
    - Doppler map의 시간 샘플링 레이트를 **8/15 Hz로 재보간(interpolate)**해서 **에포크당 16개 샘플**이 되도록 맞춤 (30초 × 8/15 Hz ≈ 16).
- 주파수축:
    - 도플러 주파수축을 **양의 주파수/음의 주파수 대역으로 분리**하여 각각 따로 학습.
    - 각 대역의 frequency bin 수: 46개.
- 정규화/노이즈 제한:
    1. 각 주파수 밴드에서, 시간축에 대해 median 값을 구함.
    2. 이 median 값들 중 최대값을 찾아, Doppler map 전체를 이 값으로 나눔 → **호흡 대역의 magnitude를 1 근처로 정규화**.
    3. 큰 움직임(자세 변경 등)에 의한 노이즈를 제한하기 위해, **값이 2를 넘는 부분은 2로 클리핑**.

이렇게 전처리된 **정규화된 Doppler map**이 이후 1D CNN + Transformer 모델의 입력으로 사용된다.

## E. Open-Database Sleep Data

---

### 어떤 데이터베이스를 썼는지

- 모델 파라미터 수가 크기 때문에, 저자들은 **대규모 공개 PSG 데이터베이스의 호흡 신호**를 함께 사용해 학습을 시켰다.
- 데이터 출처: **National Sleep Research Resources(NSRR)**에 있는 PSG들.
    - **Multi-Ethnic Study of Atherosclerosis (MESA)**
    - **Sleep Heart Health Study (SHHS)** ,
- 이들 데이터셋에는 만 명 이상 규모의 야간 PSG가 포함되어 있고, 저자들은 그 중 **흉부/복부 호흡 신호**를 사용했다.

### 어떤 신호를 썼는지

- 사용한 호흡 신호:
    - *유도 플레티스모그래피 밴드(inductive plethysmography band)**로 기록된 **흉곽(Thoracic)과 복부(Abdominal) 호흡 신호**.
    - 샘플링 레이트: **10 Hz**.
- 이 신호들로부터 **스펙트로그램(spectrogram)**을 만들었고, 이 스펙트로그램은 **UWB Doppler map과 마찬가지로 상체 움직임(호흡, 자발적 움직임)을 반영하는 time–frequency 이미지**라는 점에서 형태적으로 유사하다고 설명한다.

### 스펙트로그램 생성 방법 (모델 입력 형식 맞추기)

- 윈도우 길이: **12초**
- 슬라이딩 윈도우 간격: **15/8초** (≈ 1.875초)
    - 이렇게 설정하면 **30초(1 epoch)당 16개의 time step**이 나오도록 맞출 수 있다.
- 각 12초 구간에 대해 Fourier transform을 적용해 스펙트럼을 만들고, 이것을 시간축으로 쌓아서 **호흡 스펙트로그램**을 구성.

![스크린샷 2026-02-09 오후 4.02.50.png](%5B%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0%5DUltra-Wideband%20Radar-Based%20Sleep%20Stage%20Class/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-02-09_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.02.50.png)

- 이 스펙트로그램이 Fig. 2에서 예시로 보여주는 (a) 호흡 파형·(b) 스펙트로그램 그림과 대응된다.

### 품질 필터링(레코딩 제외 기준)

- 모든 PSG를 쓰지 않고, **호흡 스펙트로그램의 품질이 기준에 맞지 않는 레코딩은 제외**했다.
- 제외 조건:
    - 스펙트로그램에서 추출한 **호흡률(breathing rate)**이 **0.05–0.5 Hz** 범위 밖인 경우 (즉, 3–30 bpm 밖)
    - **수면 단계(label)의 변화가 밤새 지나치게 잦은 경우** (transition이 비정상적으로 많은 기록)

### 이 섹션의 역할

- 정리하면, E 섹션은:
    - UWB Doppler map과 유사한 **time–frequency 이미지 도메인(source domain)**으로
        - *“MESA/SHHS의 호흡 스펙트로그램”**을 정의하고,
    - 이후 도메인 적응(domain adaptation)에서
        - **Spectrogram (Open DB, PSG) ←→ Doppler map (UWB radar)**
            
            두 도메인 간 latent feature를 가깝게/멀게 만드는 contrastive loss를 쓸 기반 데이터를 설명한다.
            

## **F. End-to-End Deep Learning Model**

---

본 연구에서 제안한 수면 단계 분류 알고리즘은 **두 부분**으로 구성된다.

1. Doppler map으로부터 특징을 추출하는 **이미지 공간(feature extraction in image space)** 단계와,
2. 에포크 시퀀스 상의 수면 단계 변화를 학습하는 **시간 공간(temporal dynamic capture in time space)** 단계이다.

이를 위해, 먼저 **1D convolutional neural network(CNN)**에 residual connection을 적용한 구조를 사용하여, **하룻밤 Doppler map에서 N개의 연속된 에포크 구간을 잘라 입력으로 받는 세그먼트**로부터 특징을 추출하였다. 이후, 각 에포크에 해당하는 feature 시퀀스를 **Transformer encoder**에 입력하여, 수면 단계 전이의 장·단기 패턴을 모델링하였다. 전체 모델 구조는 Fig. 3에 요약되어 있다.

![스크린샷 2026-02-09 오후 4.08.02.png](%5B%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0%5DUltra-Wideband%20Radar-Based%20Sleep%20Stage%20Class/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-02-09_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.08.02.png)

---

## 1) 1D CNN 기반 특징 추출 (Image space)

입력은 전체 야간 Doppler map에서 **N 에포크 길이로 분할한 세그먼트**이며, 한 에포크는 30초로 구성된다. Doppler map은 시간–주파수 축을 갖는 이미지 형태이나, 본 연구에서는 **시간축에 초점을 맞추기 위해 2D CNN 대신 1D CNN**을 사용하였다.

- 1D CNN 블록은 **ResNet**을 변형한 구조로, 각 블록마다 **residual connection**을 포함한다.
- 시간축 방향으로 stride=2인 1D convolution을 사용하여 일부 레이어에서 **시간축 downsampling**을 수행하였고, downsampling 시 **max pooling(size=2)**을 함께 적용하여 차원을 정규화하였다.
- 첫 convolution 레이어의 필터 개수는 64개이며, 이후 일부 레이어에서 필터 수를 점진적으로 증가시켜 **최대 256개**까지 확장하였다.
- convolution의 커널 크기는 **13**으로 설정하여, 상위 레이어에서는 인접한 여러 에포크에 걸친 **국소적인 시간 의존성(inter-epoch dependency)**을 포착할 수 있도록 하였다.

네트워크는 입력 세그먼트의 길이가 N일 때, 최종적으로 **N 길이의 feature sequence를 출력하도록 설계**되었다. 즉, CNN 블록의 출력은 각 에포크에 대응하는 feature 벡터가 나열된 **[N, 256] 형태의 시퀀스**가 된다(Fig. 3 상단의 [N, 256]).

---

## 2) Transformer encoder 기반 시계열 패턴 학습 (Temporal space)

CNN에서 추출된 feature sequence [N,256][*N*,256]는 **수면 단계 전이의 장기 패턴을 학습**하기 위해 Transformer encoder 블록에 입력된다.

- Transformer는 **encoder만** 사용하며, 총 **4개의 encoder 레이어**로 구성된다.
- 각 레이어는
    - Multi-head self-attention
    - Layer normalization
    - Position-wise MLP(Feedforward 네트워크)
    - Layer normalization
        
        순서로 구성된다.
        
- 입력 시퀀스에는 **position encoding**을 더하여, 각 에포크의 상대적 위치 정보를 명시적으로 제공한다.

이러한 self-attention 기반 구조를 통해, 모델은 인접 에포크뿐 아니라 **수십~수백 에포크에 걸친 수면 단계 전이 패턴**까지 포착할 수 있게 된다.

---

## 3) 두 개의 분류 헤드: 3-class와 2-class

Transformer encoder의 출력을 활용해, 모델은 **두 가지 분류 문제를 동시에 해결**하도록 설계되었다.

1. **3-단계 수면 분류 (Wake / REM / NREM)**
    - 첫 번째 encoder 헤드는 각 에포크에 대해 **각성(Wake), REM 수면, 비REM(NREM)** 세 클래스를 분류한다.
    - 이 헤드는 Transformer encoder 출력에 fully connected layer와 softmax를 적용하여, [N,3][*N*,3] 크기의 클래스 확률을 출력한다.
2. **2-단계 NREM 세분화 (Light / Deep)**
    - 두 번째 encoder 헤드는 NREM 수면을 **가벼운 수면(Light)과 깊은 수면(Deep)**으로 나누는 이진 분류를 수행한다.
    - 이 헤드는 별도의 Transformer encoder 출력에 대해 fully connected layer와 sigmoid를 적용하여, [N,1][*N*,1] 크기의 Light/Deep 확률을 출력한다.

최종적으로, 각 에포크의 **4-단계 수면 단계(Wake, REM, Light, Deep)**는 다음과 같이 결정된다.

- 먼저 3-class 헤드에서 에포크를 **Wake, REM, NREM** 중 하나로 분류한다.
- 해당 에포크가 NREM으로 분류된 경우에 한해, 2-class 헤드의 확률을 이용해 **Light vs Deep**를 결정한다.
- 그 결과, 각 에포크는 {Wake(W), REM(R), Light(L), Deep(D)} 네 가지 클래스 중 하나의 라벨을 갖게 된다(Fig. 3의 “combination → D, L, R, W” 부분).

## F. END-TO-END DEEP LEARNING MODEL – Domain Adaptation & Loss

### 1) 왜 도메인 적응을 쓰는지

- 엔드투엔드 모델은 파라미터 수가 많아서 **UWB 레이더 데이터만으로 학습하면 데이터 부족 → overfitting** 문제가 생길 수 있다.
- 한편, Open DB(MESA/SHHS)에는 **대규모 호흡 신호 스펙트로그램**이 있고, 이건 UWB Doppler map과 **형태는 비슷하지만 도메인이 다른(source vs target)** 데이터다.
- 따라서,
    - 소스 도메인: Open DB 호흡 스펙트로그램
    - 타깃 도메인: 스마트폰 UWB Doppler map
        
        을 동시에 사용하면서, 두 도메인에서 **“같은 수면 단계라면 latent feature가 가깝고, 다르면 멀어지도록”** 학습시키는 **도메인 적응(domain adaptation)** 기법을 적용했다.
        

---

### 2) 입력 구성: spectrogram–Doppler pair

학습 시, 모델에는 매 step마다 다음과 같은 **페어(pair)**가 들어간다.

- 하나의 **호흡 스펙트로그램 세그먼트** (Open DB, MESA/SHHS)
- 하나의 **UWB Doppler map 세그먼트** (스마트폰 레이더)
- 둘 다 길이는 동일하게 **N 에포크**로 맞춘다.

각 에포크마다 두 세그먼트는 각각 **수면 단계 라벨(예: 4-stage 또는 3-stage)**을 가진다.

- 이 두 세그먼트는 서로 “동일한 사람/밤”일 필요는 없고, **랜덤 pair를 샘플링**한다.
- 중요한 건, **같은 시간 index의 에포크끼리 sleep stage label이 같은지/다른지**다.

이 pair는 CNN + Transformer를 거쳐 각각 latent feature sequence를 만든다:

- xD*xD*: Doppler map에서 나온 latent 벡터 (에포크별)
- xS*xS*: 호흡 스펙트로그램에서 나온 latent 벡터 (에포크별)

여기서 latent feature는 **Transformer 블록 이후의 intermediate output**을 사용한다.

---

### 3) Contrastive loss Lcon*Lcon*

각 에포크 t*t*에 대해, Doppler latent xD*xD*와 spectrogram latent xS*xS* 사이의 거리를 이용해서 **contrastive loss**를 정의한다.

- 먼저, 해당 에포크의 레이블이 같은지 여부를 나타내는 **distance label y*y**를 정의:
    - y=1*y*=1: 두 도메인의 수면 단계 레이블이 같음 (“close”로 표시)
    - y=0*y*=0: 레이블이 다름 (“far”)
- 두 latent 벡터 사이의 거리는 유클리드 거리 d(xD,xS)*d*(*xD*,*xS*)를 사용한다.

논문에서 정의한 contrastive loss:

$$
Lcon=yd(xD,xS)2+(1−y)Max(1−d(xD,xS),0)2
$$

직관적으로:

- 레이블이 같을 때 y=1*y*=1:
    - Lcon=d2*Lcon*=*d*2 → **거리를 0에 가깝게 만들려는 방향으로 학습** (feature를 가깝게).
- 레이블이 다를 때 y=0*y*=0:
    
    $$
    Lcon=Max(1−d,0)2Lcon=Max(1−d,0)2 →
    $$
    
    - 거리가 1보다 작으면 loss가 양수 → 더 멀어지도록 push
    - 거리가 1 이상이면 loss=0 → 충분히 떨어졌다고 보고 더 이상 당기지 않음.

결과적으로, 도메인과 상관없이 **“같은 수면 단계 = 비슷한 latent feature, 다른 수면 단계 = 일정 거리 이상 분리”**되도록 embedding space를 정렬한다.

![스크린샷 2026-02-09 오후 4.22.02.png](%5B%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0%5DUltra-Wideband%20Radar-Based%20Sleep%20Stage%20Class/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-02-09_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.22.02.png)

---

## 4) Classification loss Lcls*Lcls*

동시에, 각 도메인(스펙트로그램, Doppler)에서의 **수면 단계 분류 정확도**를 높이기 위해 일반적인 **class-weighted cross-entropy loss Lcls*Lcls***를 사용한다.

- 이유: 4-stage 분류에서 light sleep이 전체의 50% 이상, REM/Deep은 25% 미만이라 **클래스 불균형**이 심함.
- 클래스별 weight는 **각 클래스의 샘플 수에 반비례**하도록 설정해서,
    - 소수 클래스(REM, Deep)의 오차가 loss에 더 크게 반영되도록 한다.

Lcls*Lcls*는 3-class/2-class 헤드에서 나오는 예측과 ground truth 레이블을 모두 포함하는 형태라고 보면 된다.

---

## 5) 전체 손실 Ltotal*Ltotal* 및 학습 스케줄

전체 학습에서는 두 loss를 가중합으로 사용한다:

$$
Ltotal=αLcls+βLcon
$$

- α, β는 hyperparameter로,
    - classification 성능과 도메인 정렬(domain alignment)의 균형을 조절하는 역할.

학습 스텝은 크게 두 단계로 나뉜다.

1. **도메인 적응 (pre-training)**
    - 입력: Open DB 호흡 스펙트로그램 + UWB Doppler map pair.
    - 목적:
        - $Lcls$: 두 도메인 각각에서 수면 단계 분류 학습.
        - $Lcon$: 두 도메인의 latent feature space를 “sleep-stage-aware”하게 정렬.
2. **UWB 전용 파인튜닝 (fine-tuning)**
    - 이후, **UWB Doppler map만** 사용해서 **낮은 learning rate(1e-5)**로 미세 조정.
    - 이 단계에서는 사실상 **타깃 도메인(UWB)에 특화된 세부 튜닝**을 수행하는 것.

---

## 6) Inference 파이프라인 (Fig. 5 요약)

추론 시에는:

- 전체 야간 Doppler map에서 **길이 N 에포크 세그먼트**를 1 에포크씩 슬라이딩하면서 잘라서 입력.
- 양·음의 도플러 주파수 대역에 대해 각각 모델 추론 후, **두 대역의 확률을 평균**.
- 서로 겹치는 세그먼트에서 나온 에포크별 확률도 **겹치는 부분끼리 평균**해 최종 에포크별 sleep stage 확률을 얻는다.
- 3-class(W/REM/NREM) + 2-class(Light/Deep) 결과를 조합해 4-stage 라벨을 결정.

![스크린샷 2026-02-09 오후 4.22.40.png](%5B%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0%5DUltra-Wideband%20Radar-Based%20Sleep%20Stage%20Class/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-02-09_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.22.40.png)

## G. VALIDATION

---

<데이터 분할 및 검증 설계>

제안한 알고리즘의 성능을 공정하게 평가하기 위해, 내부 검증 데이터셋(KOSLEEP + Instituto do Sono)을 대상으로 **5-fold cross-validation**을 수행했다. 두 기관의 전체 야간 기록을 5개 fold로 나누고, 매 반복마다 4개 fold를 학습·검증용으로, 나머지 1개 fold를 테스트용으로 사용했다. 각 fold에서 테스트에 사용된 에포크들을 모두 합쳐 최종 성능 지표를 계산한다.

학습 데이터에는 **탁자 위 스마트폰 세팅과 삼각대(tripod) 세팅에서 얻은 UWB 데이터가 모두 포함**되지만, 테스트 데이터는 **실제 사용 시나리오에 해당하는 “침대 옆 탁자에 둔 스마트폰” 데이터만 사용**했다. 즉, 모델은 보다 이상적인 tripod 환경까지 포함해 풍부한 패턴을 학습하되, 최종 평가는 현실적인 배치 조건에서만 수행하도록 설계되었다.

각 fold마다 대략적으로

- KOSLEEP: 88명(학습/검증), 22명(3박 테스트)
- Instituto do Sono: 144명(학습/검증), 36명(1박 테스트)
    
    으로 나뉘며, 총 102박의 야간 수면 기록이 각 fold의 테스트셋으로 사용된다.
    

또한, 모델의 일반화 성능을 평가하기 위해 별도로 모집한 **삼성서울병원(SMC) 56명** 데이터에 대해 **external validation**을 수행했다. 이 데이터는 내부 5-fold 학습에 사용되지 않고, 학습이 끝난 모델에 그대로 적용하여 추가적인 외부 검증 성능을 보고한다.

## H. EVALUATION METRIC

---

<왜 Accuracy만 쓰면 안 되는지>

수면 단계 데이터는 클래스 불균형이 심한 전형적인 상황이다. **Light sleep**이 전체 수면의 50% 이상을 차지하는 반면, **REM 및 Deep sleep은 각각 25% 미만**이기 때문에, 모든 에포크를 Light sleep으로만 예측해도 50% 이상의 Accuracy가 나올 수 있다. 따라서 단순 Accuracy는 모델의 진짜 성능을 제대로 반영하지 못한다.

이에 따라, 저자들은 다음 네 가지 정량 지표를 사용했다.

1. **Accuracy**
    - 전체 에포크 중 예측 라벨이 PSG 라벨과 일치하는 비율.
2. **Cohen’s kappa**
    - 우연 일치를 보정한 agreement 지표로, **클래스 불균형에 더 강건**하다.
    - 일반적으로
        - kappa > 0.4: moderate
        - kappa > 0.6: substantial
        - kappa > 0.8: almost perfect agreement로 해석한다.
3. **Balanced Accuracy (BA)**
    - 각 클래스의 sensitivity(=recall)를 계산한 뒤, **평균을 낸 값**.
    - Light/Deep처럼 데이터가 적은 클래스가 전체 성능에 충분히 반영되도록 돕는다.
4. **Macro F1-score**
    - 클래스별 F1을 산출하고, 그 평균을 계산.
    - precision과 recall을 함께 고려하여, 전체적인 분류 품질을 평가한다.

추가로, 각 참가자 단위로 계산한 **수면 임상 지표**(TST, WASO, SOL, SE, REM latency, 각 stage 비율)에 대해서는,

- **Mean Absolute Difference (MAD)**: PSG 대비 절대 오차의 평균
- **표준편차(SD)**: 오차의 분산
- **Intra-class Correlation (ICC)**: PSG와 모델 간 agreement

를 함께 보고한다. ICC는

- 0.5 이상: moderate,
- 0.75 이상: good,
- 0.90 이상: excellent reliability로 해석한다.

## **III. RESULTS**

---

## A. Model Performance

<에포크 단위 수면 단계 분류 성능>

Table 3에는, 509명의 모든 참가자에 대해 **에포크 단위로 PSG 라벨과 모델 예측을 비교한 결과**가 정리되어 있다. 입력 세그먼트 길이는 240 에포크(약 120분)일 때의 성능이다.

![스크린샷 2026-02-09 오후 4.53.56.png](%5B%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0%5DUltra-Wideband%20Radar-Based%20Sleep%20Stage%20Class/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-02-09_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.53.56.png)

핵심 숫자는 다음과 같다.

- **4-stage 분류 (Wake, REM, Light, Deep)** – 전체 참가자(509명):
    - Accuracy: 0.766
    - Cohen’s kappa: 0.642
    - Balanced accuracy: 0.747
    - Macro F1: 0.733
- **3-stage 분류 (Wake, REM, NREM)** – 전체 참가자:
    - Accuracy: 0.855
    - kappa: 0.735
    - BA: 0.849
    - Macro F1: 0.828
- **2-stage 분류 (Wake vs Sleep)** – 전체 참가자:
    - Accuracy: 0.913
    - kappa: 0.731
    - BA: 0.882
    - Macro F1: 0.866

4-stage보다 3-stage/2-stage로 stage를 통합했을 때 kappa와 Accuracy가 전반적으로 상승하며, 특히 **Wake vs Sleep 분류에서는 ACC 0.91** 수준으로 매우 높은 정확도를 보인다.

또한, 수면무호흡 정도(AHI)에 따라 참가자를 네 그룹으로 나누어 4-stage 성능을 비교하면, **정상(AHI<5)과 환자(AHI≥5) 그룹 모두에서 kappa가 약 0.64 수준으로 거의 동일**하다. 이는 모델이 정상인뿐 아니라 수면무호흡 환자에서도 **공통적인 패턴을 잘 학습해 일반**

---

<클래스별 혼동 패턴: Confusion Matrix>

Fig. 7은 전체 참가자를 대상으로 한 **4-stage 혼동행렬(confusion matrix)**을 보여준다. 각 행은 PSG 기준 참 라벨, 각 열은 모델 예측 라벨이며, **행 기준 정규화(sensitivity)**로 나타냈다.

![스크린샷 2026-02-09 오후 4.54.32.png](%5B%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0%5DUltra-Wideband%20Radar-Based%20Sleep%20Stage%20Class/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-02-09_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.54.32.png)

주요 관찰 사항은 다음과 같다.

- Wake: sensitivity ≈ 0.831 (WAKE를 WAKE로 맞출 확률이 가장 높음).
- REM: sensitivity ≈ 0.854.
- Light: sensitivity ≈ 0.758.
- Deep: sensitivity ≈ 0.545 (가장 낮은 stage).

가장 큰 오분류는 **Deep ↔ Light 사이**에서 발생하며, Deep 에포크의 약 42.4%가 Light로 잘못 분류된다. 이는 원래 수면다원검사에서도 Deep과 Light 구분이 annotator 간 agreement가 가장 낮은 stage라는 점과 맞닿아 있다(논문 Discussion에서 별도 언급).

---

<대표 Hypnogram 예시>

Fig. 8은 한 참가자에 대한 **대표 hypnogram**으로, 전문가 PSG 라벨과 모델 예측 라벨을 시간 축에 따라 나란히 표시한 예시다. 이 참가자의 성능은 kappa 0.667, Accuracy 0.784 정도로, 전체 참가자 중 “평균에 가까운” 케이스이다.

![스크린샷 2026-02-09 오후 4.55.00.png](%5B%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0%5DUltra-Wideband%20Radar-Based%20Sleep%20Stage%20Class/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-02-09_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.55.00.png)

그림을 보면,

- 전체적인 수면 구조(REM/NREM 교대, 기상/각성 구간의 위치)는 PSG와 상당히 유사하며,
- Deep과 Light 간 경계에서 약간의 mismatch가 있지만,
- 주요 각성 구간과 REM 에피소드의 위치는 꽤 잘 맞는 것을 확인할 수 있다.

## III.B Additional Analysis

## 1) External validation (SMC 데이터셋)

내부 5-fold 검증 외에도, 제안한 모델의 일반화 성능을 확인하기 위해 **삼성서울병원(SMC) 56명** 데이터셋에 대한 외부 검증을 수행했다. 이 데이터는 모델 학습에 전혀 사용되지 않았고, 학습이 완료된 모델에 그대로 적용되었다.

SMC 외부 검증에서 4-stage 분류 성능은 다음과 같다(Table 3 마지막 행).

- Kappa: 0.635
- Accuracy: 0.781
- Balanced accuracy: 0.727
- Macro F1: 0.689

내부 5-fold 결과(All 509: kappa 0.642, ACC 0.766)와 비교하면, **외부 데이터셋에서도 유사한 수준의 kappa(0.635)를 유지**하며 Accuracy는 0.78 정도로 오히려 약간 증가하는 모습을 보인다. 전체적으로는 성능이 약간 감소했지만, 여전히 **“substantial agreement” 구간(kappa>0.6)을 유지**하고 있어, 서로 다른 기관에서 수집된 데이터에도 모델이 어느 정도 신뢰성 있게 적용될 수 있음을 보여준다.

---

## 2) Domain adaptation 효과 분석 (Fig. 9)

저자들은 제안한 **도메인 적응 기반 학습 방법**이 실제로 성능 향상에 기여하는지 확인하기 위해, 여러 학습 전략을 비교했다. Fig. 9에는 다음 네 가지 전략의 Cohen’s kappa가, 사용한 UWB 학습 데이터 비율(10%, 50%, 100%)에 따라 비교되어 있다.

1. **UWB only (from scratch)**
    - UWB Doppler 데이터만으로 처음부터 학습.
2. **Conventional transfer learning**
    - Open DB 호흡 스펙트로그램으로 먼저 pre-training 후,
    - UWB 데이터로 특정 레이어만 fine-tune:
        - CNN만 / Transformer만 / Fully Connected만 / 전체 레이어.
3. **Proposed domain adaptation**
    - Open DB 스펙트로그램 + UWB Doppler를 동시에 입력
    - Lcls+Lcon*Lcls*+*Lcon*으로 도메인 정렬하면서 학습.

![스크린샷 2026-02-09 오후 4.56.20.png](%5B%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0%5DUltra-Wideband%20Radar-Based%20Sleep%20Stage%20Class/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-02-09_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.56.20.png)

주요 결과는 다음과 같다.

- **UWB 데이터가 10%만 있을 때**,
    - UWB only vs Proposed의 kappa 차이가 약 **0.07**로, 도메인 적응이 매우 큰 이득을 준다.
- UWB 데이터가 50%, 100%로 늘어날수록,
    - 두 방법 간 차이는 줄어들지만, **전체 데이터(100%)를 사용한 경우에도 제안 방법이 kappa 약 0.01 정도 더 높음**.
- 일반적인 transfer learning(호흡 스펙트로그램 pretrain 후 부분 fine-tuning)은,
    - 적은 데이터(10%)에서는 제안 방법과 비슷하지만
    - 데이터가 많아질수록 제안한 도메인 적응이 더 높은 성능을 보인다.
- 특히 **FC 레이어만 fine-tune하는 방식은 데이터가 늘어나도 성능이 거의 증가하지 않아**, feature representation 자체를 업데이트하지 않으면 이득이 제한적이라는 점을 보여준다.

요약하면, **레이더+PSG 데이터를 크게 늘리기 어려운 상황에서 Open DB 호흡 신호를 도메인 적응 방식으로 활용하는 것이 특히 유리**하고, 데이터가 많아진 뒤에도 여전히 약간의 성능 이득을 유지한다.

---

## 3) Epoch 길이와 모델 구조 ablation (Table 5)

추가 분석으로, 저자들은 **입력 세그먼트 길이(에포크 수)**와 **모델 구성**에 따라 성능이 어떻게 달라지는지도 평가했다. 비교 대상은 다음 세 가지 모델이다(Fig. 6 참고).

1. **Variant 1: CNN-only 모델**
    - 제안된 1D CNN 블록만 사용, Transformer 없음.
2. **Variant 2: Epoch-wise CNN + Transformer**
    - CNN은 각 에포크별로 독립 feature만 추출 (inter-epoch 정보 없음).
    - 이후 Transformer가 이 feature sequence를 받아 temporal dependency를 학습.
3. **Proposed model (1D CNN + Transformer)**
    - CNN이 layer가 깊어질수록 **시간축을 축소하면서 인접 에포크 사이의 dependency까지 일부 반영**하고,
    - Transformer가 이를 기반으로 더 긴 범위의 temporal dynamics를 학습.

에포크 길이는 30, 60, 120, 240, 360 에포크로 바꿔 가며 kappa를 비교했다.

> **[Epoch-length별 세 모델의 Cohen’s kappa]**
> 
> 
> ![스크린샷 2026-02-09 오후 4.56.48.png](%5B%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0%5DUltra-Wideband%20Radar-Based%20Sleep%20Stage%20Class/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-02-09_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.56.48.png)
> 

핵심 결과는 다음과 같다.

- **모든 모델에서 epoch 길이를 240까지 늘리면 성능이 꾸준히 향상**된다.
    - 수면 단계 변화가 보통 90–120분 주기의 sleep cycle로 반복되기 때문에,
    - 240 epoch(약 120분) 길이가 하나 이상의 cycle 패턴을 담기에 적절하다는 해석이 가능하다.
- Proposed model은 모든 epoch 길이에서 두 variant보다 **유의하게 높은 kappa**를 기록한다.
    - 예를 들어 epoch=240 기준으로,
        - Proposed vs CNN-only: kappa 차이 ≈ +0.06
        - Proposed vs Epoch-wise CNN+Transformer: 차이 ≈ +0.07 이상.
- epoch 길이를 360으로 더 늘렸을 때는, **Proposed 모델의 성능이 더 이상 증가하지 않고 plateau**를 보인다.
    - 이는 너무 긴 시퀀스는 계산 비용과 latency만 늘리고, 추가적인 성능 이득은 제한적일 수 있음을 시사한다.

이 ablation 결과는,

- 단순히 Transformer만 붙이는 것이 아니라,
- **CNN에서부터 inter-epoch dependency를 일부 반영하고 Transformer로 long-range 정보를 보완하는 현재 구조가 타협점이 좋다**는 것을 뒷받침한다.
    
    또한, 수면 단계 변화가 상대적으로 느리게 일어나기 때문에 **context window를 어느 정도 길게 잡는 것이 유의미하다**는 실험적 근거를 제공한다.
    

## V. Discussion – 저자 주장 요약

---

저자들은 본 연구가 **“스마트폰 + UWB 레이더만으로 4-stage 수면 분류를 상용 기기 수준으로 달성했다”**는 점을 가장 큰 기여로 강조한다. 기존 마이크 기반 스마트폰 방법들은 수면 단계 분류에서 만족스러운 성능을 얻지 못했는데, 이는 소리로는 몸의 세밀한 움직임을 충분히 포착하기 어렵기 때문이라고 지적한다. 반면, 이 연구에서는 **침대 옆 탁자에 둔 스마트폰의 레이더만으로도 kappa≈0.64, ACC≈0.76의 4-stage 성능**을 보여, 여러 상용 non-contact 기기(Table 6)와 비슷하거나 더 나은 수준이라고 주장한다.

또한, 기존 레이더 기반 연구들 중 일부는 더 높은 성능을 보고했지만, 대부분 **레이더 위치/각도를 엄격히 제어한 실험실 환경(삼각대, 벽 설치, 최적 거리·높이)**에 의존한 결과라는 점을 짚는다. 이 논문은 **“현실적인 침대 옆 탁자” 배치**를 기준으로 성능을 평가했다는 점에서, 실사용 관점의 의미가 크다고 본다. 실제로 tripod 데이터로만 평가하면 kappa가 0.69까지 올라가, 기존 최고 성능 연구들과 비교해도 경쟁력이 있음을 언급한다.

수면무호흡증 환자 포함 여부도 중요한 포인트다. 무호흡 환자는 각성 증가, deep sleep 감소 등 전형적인 수면 구조 이상을 보이기 때문에, 모델이 이들을 포함한 데이터에서 학습·검증되었다는 점을 강조한다. internal dataset에서 AHI<5 vs AHI≥5 그룹 간 kappa 차이가 0.01 수준에 불과해, **정상군과 무호흡군 모두에서 유사한 수준의 성능**을 보였다는 점을 “robustness”의 근거로 제시한다.

모델 구조에 관해서는,

- CNN이 에포크 간 local dependency를 일부 반영하고,
- Transformer가 long-range temporal dynamics를 보완하는 구조,
- epoch-length를 240(약 120분)까지 늘렸을 때 성능이 크게 좋아지는 점
    
    등을 묶어서, **수면 사이클(90–120분) 패턴을 포착할 수 있는 충분한 context window가 중요하다**고 해석한다.
    

마지막으로, 도메인 적응 측면에서, **Open DB 호흡 스펙트로그램과 UWB Doppler를 함께 사용하는 contrastive learning 방식이, 단순 UWB-only 학습이나 전통적인 transfer learning보다 일관되게 더 좋은 성능을 준다**고 정리한다. 특히 UWB 데이터가 적을 때(10%) 성능 차이가 크게 나며, 전체 데이터(100%)에서도 여전히 약간의 이득을 유지한다는 점을 강조한다.

---

## V. Limitations – 저자 언급 한계

저자들이 명시한 주요 한계는 세 가지다.

1. **안테나/하드웨어 설계의 제약**
    - 스마트폰 설계 시점에 레이더를 고려한 구조가 아니어서, TX/RX 안테나를 이상적인 방향(둘 다 피험자 정면)으로 배치하지 못했다.
    - 현재는 스마트폰 기능을 해치지 않는 선에서 측면/뒷면에 안테나를 배치했다 보니, **신호 품질이 최적이 아니며, 제조 단계에서 안테나를 최적 설계하면 성능 향상 여지가 있다**고 본다.
2. **Deep sleep sensitivity가 낮음**
    - Deep vs Light 구분에서 sensitivity가 가장 낮다(Deep sensitivity≈0.545).
    - 원인 1: PSG 전문가 간 deep stage 라벨 agreement 자체가 가장 낮은 stage임 (약 67.4%, 주로 Light와 혼동).
    - 원인 2: Deep sleep 구분에 중요한 HR 패턴이 레이더 신호 품질 한계 때문에 충분히 포착되지 못했을 가능성.
    - 스마트폰 배치 제약을 희생하고 신호 품질을 더 끌어올리면 deep 분류는 개선될 수 있지만, 사용자 편의성이 떨어질 수 있다는 trade-off를 인정한다.
3. **현실 세계 사용 시나리오의 다양성 미완전**
    - 침대 옆 탁자 높이가 매우 낮거나 높은 경우, 스마트폰을 멀리 두는 경우 등 다양한 실제 사용 환경을 모두 커버하지는 못했다.
    - 레이더 특성상 **거리/각도에 따라 SNR이 달라지는 문제**가 있기 때문에, 실제 서비스에서는 사용자에게 “어느 정도의 배치 가이드”를 제공해야 한다고 제안한다.
4. **동침(두 명 이상 같이 자는) 상황 미고려**
    - 실험은 한 명만 침대에 있는 상황만 가정했다.
    - 레이더로는 거리 차이를 이용해 여러 사람의 신호를 분리하는 것이 이론적으로 가능하다고 보고, 추후 연구에서 multi-person scenario를 다루겠다고 남겨 둔다.

---

## VI. Conclusion – 저자 결론 요약

결론에서 저자들은 다음을 요약한다.

- 1D CNN + Transformer로 구성된 엔드투엔드 모델과 도메인 적응 기법을 통해, **스마트폰 UWB 레이더만으로 4-stage 수면 분류를 substantial 수준(kappa≈0.64)까지 달성**했다.
- 500명 이상, 다양한 무호흡 중증도를 포함하는 대규모 데이터를 기반으로 성능을 검증했고, 외부 검증에서도 유사한 지표를 유지했다.
- “스마트폰을 침대 옆 탁자에 두기만 하면 되는” 구성이라, **추가 기기 없이도 높은 사용성과 접근성을 제공하는 수면 모니터링 솔루션**이 될 수 있다고 주장한다.

---

## 내 시각: 장점, 한계, 네 연구와의 연결 포인트

## 1) 장점 – 왜 이 논문이 재밌는지

- **현실적인 폼팩터 + 대규모 데이터**
    - 스마트폰이라는 가장 보편적인 디바이스를 target으로, 509박 + 외부 56명까지 확보한 게 인상적이다.
    - 네가 하고 있는 FSR/PPG 기반 수면 분류도, “추가 기기를 최소화한 HCI/폼팩터”를 고민하는 입장에서 참고할 만한 설계 철학.
- **도메인 적응을 실제 의료 시나리오에 잘 녹인 사례**
    - NSRR(MESA/SHHS)의 대규모 PSG 호흡 데이터를 “source domain”으로 쓰고, 레이더를 target으로 맞추는 구성이 깔끔하다.
    - 단순 transfer learning이 아니라, **epoch-wise sleep stage label을 활용한 contrastive loss로 embedding space를 align**한 점이 특히 재사용 가치가 높음.
- **모델 구조 설계가 수면 physiology와 잘 맞는다**
    - epoch-length 240(≈120분)에서 성능이 최대가 되는 점을 보고 “sleep cycle 길이에 맞춘 context window”라는 해석을 붙인 게 설득력 있다.
    - 1D CNN에서 local inter-epoch dep를 일부 보고, Transformer에서 long-range를 본다는 decomposition도 수면 단계 시퀀스 특성에 잘 맞는다.

---

## 2) 한계 – 너 입장에서 볼 때 아쉬운 점

- **Raw signal 공개/재현 가능성이 낮음**
    - 삼성 커스터마이즈 스마트폰 + IR-UWB라는 특수 HW라, 그대로 재현하기는 어렵다. (네 연구에 직접적인 엔드투엔드 replication은 힘듦)
- **도메인 적응 설계의 디테일이 일부 추상적**
    - pair sampling 전략(예: label 매칭 비율, positive/negative 비율), α,β*α*,*β* 설정, embedding dimension 등 practical hyperparameter 세팅은 논문에서는 깊게 안 들어간다.
    - 실제로 적용할 땐 이쪽을 네 환경(PPG/FSR)에서 튜닝해야 할듯.
- **deep sleep sensitivity 개선 전략은 Discussion에서만 제안 수준**
    - 추가적으로 heart-rate-like 정보를 더 잘 뽑는 방법(예: higher freq resolution, multi-band modeling)에 대한 concrete 제안은 없다.

---

## 3) 네 연구와의 연결 포인트

네가 하고 있는 **FSR/PPG 기반 수면 단계 분류**에 이 논문에서 가져갈 수 있는 아이디어는 꽤 많다:

1. **Open DB + 도메인 적응 패턴**
    - 예:
        - Source: 공개 PPG + EEG 기반 PSG dataset (MESA, SHHS, MASS 등)에서의 PPG 스펙트로그램 / 특징.
        - Target: 네가 실제로 수집한 PPG/FSR 또는 wearable-like 센서.
    - 동일 epoch-wise sleep stage 라벨을 써서, **“같은 stage면 latent를 가깝게, 다르면 멀리”** 하는 contrastive loss를 똑같이 쓸 수 있다.
2. **입력 길이(컨텍스트) 설계**
    - 현재 모델이 염두에 두는 context 길이(예: 30분, 60분, 120분)를 이 논문처럼 **실험적으로 바꿔 보면서 최적인 길이를 찾는 ablation**을 해 볼 수 있다.
    - 특히 네 연구에서도 “한 번에 30~60분 시퀀스를 넣고 multi-epoch classification”을 하는 쪽으로 구조를 바꾸면, 성능 jump가 있을 수 있음.
3. **모델 구조: CNN + Transformer의 역할 분리**
    - 지금 네가 쓰는 LSTM/Transformer 기반 구조를,
        - “시간축에 특화된 1D CNN(feature extractor) + Transformer(context modeler)”로 리팩터링하는 것도 고려해볼 만하다.
    - 특히, inter-epoch dependency를 CNN의 stride/pooling 설계로 일부 encode하는 아이디어는, HRV/호흡 패턴을 반영할 때도 쓸 수 있는 패턴.
4. **임상 파라미터 측면 평가**
    - 이 논문처럼 TST, WASO, SE, Stage 분포, REM latency에 대한 MAD/ICC까지 보는 평가 세트를 네 연구에도 도입하면,
        - “단순 epoch-level 분류 정확도”를 넘어 “임상적으로 쓸 만한 수준인지”까지 어필하기 좋을 것 같다.

---

## 4) 실제 구현/응용 때 고려할 점

- **데이터 부족 + 도메인 간 차이**
    - 네 환경에서도 source/target 간 센서 특성, 샘플링 레이트, 노이즈가 다르기 때문에,
    - 이 논문과 같이 **domain adaptation + class imbalance handling**을 세트로 가져오는 게 중요해 보인다.
- **Context 길이 vs 메모리/latency 트레이드오프**
    - 240 epoch(120분)는 성능에 유리하지만, 실제 실시간 inference에서는 latency/메모리 부담이 크다.
    - 네 use case(실시간 피드백인지, 수면 후 리포트인지)에 맞춰 60–120분 정도에서 타협점을 찾을 필요가 있다.
- **실제 배치 환경 다양성**
    - 이 논문처럼 “현실적인 배치에서 성능이 어떻게 깨지는지”에 대한 실험을 일부라도 해 보는 게 좋다.
    - 예: FSR 센서 위치 편차, 손목 밴드 착용 느슨함, PPG SNR 변화 등의 robustness 실험.