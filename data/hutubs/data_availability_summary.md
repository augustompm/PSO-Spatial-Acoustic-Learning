# HUTUBS Database - Resumo de Disponibilidade de Dados

## Visão Geral
- **Total de sujeitos**: 96
- **Arquivos SOFA (HRTFs)**: 90 sujeitos
- **Medidas antropométricas completas**: 93 sujeitos
- **Meshes 3D disponíveis**: 58 sujeitos

## Sujeitos com Dados Faltantes

### Sem arquivos SOFA
- Sujeitos: 18, 56, 79, 80, 92, 94

### Sem medidas antropométricas
- Sujeitos: 18, 79, 92 (todas as medidas faltando)

### Sobreposição
- Sujeitos 18, 79, 92: Sem SOFA E sem medidas antropométricas
- Sujeitos 56, 80, 94: Sem SOFA mas COM medidas antropométricas

## Estrutura das Medidas Antropométricas

### Medidas da Cabeça e Corpo (x1-x17)
- **x1**: head width (largura da cabeça)
- **x2**: head height (altura da cabeça) 
- **x3**: head depth (profundidade da cabeça)
- **x4**: pinna offset down (deslocamento vertical da orelha)
- **x5**: pinna offset back (deslocamento posterior da orelha)
- **x6**: neck width (largura do pescoço)
- **x7**: neck height (altura do pescoço)
- **x8**: neck depth (profundidade do pescoço)
- **x9**: torso top width (largura superior do torso)
- **x12**: shoulder width (largura dos ombros)
- **x14**: height (altura total/estatura)
- **x16**: head circumference (circunferência da cabeça)
- **x17**: shoulder circumference (circunferência dos ombros)

### Medidas da Orelha (d1-d10, theta1-theta2)
Para cada orelha (L=esquerda, R=direita):
- **d1**: cavum concha height
- **d2**: cymba concha height
- **d3**: cavum concha width
- **d4**: fossa height
- **d5**: pinna height
- **d6**: pinna width
- **d7**: intertragal incisure
- **d8**: cavum concha depth (down)
- **d9**: cavum concha depth (back)
- **d10**: crus of helix depth
- **theta1**: pinna rotation angle (graus)
- **theta2**: pinna flare angle (graus)

## Notas Importantes

1. **Consentimento**: Nem todos os sujeitos concordaram em ter todas as medidas tomadas
2. **Meshes 3D**: Apenas 58 dos 96 sujeitos têm meshes 3D disponíveis
3. **Unidades**: 
   - Distâncias em centímetros
   - Ângulos em graus
4. **Dados completos**: 93 sujeitos têm todas as medidas antropométricas
5. **Extração de meshes**: As meshes contêm apenas a cabeça, não o corpo completo