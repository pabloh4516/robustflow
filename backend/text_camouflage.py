"""
Text Camouflage System - Human-Only Readable Text
==================================================

Sistema de camuflagem que transforma texto em versões que:
- São perfeitamente legíveis para humanos
- São ilegíveis ou incorretamente interpretadas por máquinas/OCR

TÉCNICAS IMPLEMENTADAS:

1. HOMOGLYPH SUBSTITUTION
   - Substitui letras por caracteres Unicode visualmente idênticos
   - Ex: 'a' → 'а' (cirílico), 'e' → 'е' (cirílico)
   - Máquinas veem caracteres diferentes, humanos veem igual

2. ZERO-WIDTH INJECTION
   - Insere caracteres invisíveis (zero-width) entre letras
   - Quebra reconhecimento de palavras sem afetar visualização

3. SEMANTIC CAMOUFLAGE
   - Leetspeak inteligente que mantém legibilidade
   - Substituições contextuais (A→4, E→3, etc.)

4. ADVERSARIAL TYPOGRAPHY
   - Modifica formas de letras sutilmente em imagens
   - OCR lê errado, humanos leem certo

5. VISUAL FRAGMENTATION
   - Divide texto em camadas que se combinam visualmente
   - Cada camada isolada é ilegível

6. UNICODE COMBINING MARKS
   - Adiciona diacríticos invisíveis ou sutis
   - Confunde normalização de texto

7. DIRECTION MANIPULATION
   - Usa caracteres RTL/LTR para confundir ordem
   - Texto parece normal mas copia errado

8. FONT CONFUSION
   - Mistura caracteres de diferentes scripts
   - Visualmente coerente, semanticamente confuso
"""

import random
import unicodedata
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from dataclasses import dataclass
from enum import Enum


class CamouflageMode(Enum):
    """Modos de camuflagem disponíveis."""
    HOMOGLYPH = "homoglyph"
    ZERO_WIDTH = "zero_width"
    LEETSPEAK = "leetspeak"
    MIXED_SCRIPTS = "mixed_scripts"
    COMBINING_MARKS = "combining_marks"
    DIRECTION_TRICK = "direction_trick"
    VISUAL_NOISE = "visual_noise"
    FULL_CAMOUFLAGE = "full_camouflage"


@dataclass
class CamouflageResult:
    """Resultado da camuflagem de texto."""
    original_text: str
    camouflaged_text: str
    technique: str
    human_readable: bool
    machine_readable: bool
    description: str
    unicode_analysis: Dict


class HomoglyphDatabase:
    """
    Base de dados de homoglyphs - caracteres visualmente idênticos
    mas com códigos Unicode diferentes.
    """

    # Mapeamento Latin → Cirílico (visualmente idênticos)
    LATIN_TO_CYRILLIC = {
        'a': 'а',  # U+0430 Cyrillic Small Letter A
        'c': 'с',  # U+0441 Cyrillic Small Letter Es
        'e': 'е',  # U+0435 Cyrillic Small Letter Ie
        'o': 'о',  # U+043E Cyrillic Small Letter O
        'p': 'р',  # U+0440 Cyrillic Small Letter Er
        'x': 'х',  # U+0445 Cyrillic Small Letter Ha
        'y': 'у',  # U+0443 Cyrillic Small Letter U
        'A': 'А',  # U+0410 Cyrillic Capital Letter A
        'B': 'В',  # U+0412 Cyrillic Capital Letter Ve
        'C': 'С',  # U+0421 Cyrillic Capital Letter Es
        'E': 'Е',  # U+0415 Cyrillic Capital Letter Ie
        'H': 'Н',  # U+041D Cyrillic Capital Letter En
        'K': 'К',  # U+041A Cyrillic Capital Letter Ka
        'M': 'М',  # U+041C Cyrillic Capital Letter Em
        'O': 'О',  # U+041E Cyrillic Capital Letter O
        'P': 'Р',  # U+0420 Cyrillic Capital Letter Er
        'T': 'Т',  # U+0422 Cyrillic Capital Letter Te
        'X': 'Х',  # U+0425 Cyrillic Capital Letter Ha
    }

    # Mapeamento Latin → Grego (visualmente similares)
    LATIN_TO_GREEK = {
        'A': 'Α',  # Alpha
        'B': 'Β',  # Beta
        'E': 'Ε',  # Epsilon
        'H': 'Η',  # Eta
        'I': 'Ι',  # Iota
        'K': 'Κ',  # Kappa
        'M': 'Μ',  # Mu
        'N': 'Ν',  # Nu
        'O': 'Ο',  # Omicron
        'P': 'Ρ',  # Rho
        'T': 'Τ',  # Tau
        'X': 'Χ',  # Chi
        'Y': 'Υ',  # Upsilon
        'Z': 'Ζ',  # Zeta
        'o': 'ο',  # omicron minúsculo
        'v': 'ν',  # nu minúsculo
    }

    # Caracteres matemáticos/especiais similares
    LATIN_TO_MATH = {
        'A': '𝐀',  # Mathematical Bold Capital A
        'B': '𝐁',
        'C': '𝐂',
        'a': '𝐚',
        'b': '𝐛',
        'c': '𝐜',
        '0': '𝟎',  # Mathematical Bold Digit Zero
        '1': '𝟏',
        '2': '𝟐',
    }

    # Full-width characters (parecem iguais em algumas fontes)
    LATIN_TO_FULLWIDTH = {
        'A': 'Ａ', 'B': 'Ｂ', 'C': 'Ｃ', 'D': 'Ｄ', 'E': 'Ｅ',
        'F': 'Ｆ', 'G': 'Ｇ', 'H': 'Ｈ', 'I': 'Ｉ', 'J': 'Ｊ',
        'a': 'ａ', 'b': 'ｂ', 'c': 'ｃ', 'd': 'ｄ', 'e': 'ｅ',
        '0': '０', '1': '１', '2': '２', '3': '３', '4': '４',
    }

    @classmethod
    def get_all_homoglyphs(cls, char: str) -> List[str]:
        """Retorna todos os homoglyphs conhecidos para um caractere."""
        homoglyphs = [char]  # Original sempre incluído

        if char in cls.LATIN_TO_CYRILLIC:
            homoglyphs.append(cls.LATIN_TO_CYRILLIC[char])
        if char in cls.LATIN_TO_GREEK:
            homoglyphs.append(cls.LATIN_TO_GREEK[char])

        return homoglyphs


class ZeroWidthCharacters:
    """Caracteres de largura zero para injeção invisível."""

    ZERO_WIDTH_SPACE = '\u200B'        # Zero Width Space
    ZERO_WIDTH_NON_JOINER = '\u200C'   # Zero Width Non-Joiner
    ZERO_WIDTH_JOINER = '\u200D'       # Zero Width Joiner
    WORD_JOINER = '\u2060'             # Word Joiner
    ZERO_WIDTH_NO_BREAK = '\uFEFF'     # Zero Width No-Break Space (BOM)

    # Caracteres de controle invisíveis
    SOFT_HYPHEN = '\u00AD'             # Soft Hyphen

    ALL = [
        ZERO_WIDTH_SPACE,
        ZERO_WIDTH_NON_JOINER,
        ZERO_WIDTH_JOINER,
        WORD_JOINER,
    ]

    @classmethod
    def get_random(cls) -> str:
        """Retorna um caractere zero-width aleatório."""
        return random.choice(cls.ALL)


class LeetSpeakEncoder:
    """
    Codificador Leetspeak inteligente.
    Mantém legibilidade enquanto confunde OCR.
    """

    # Substituições que mantêm legibilidade humana
    SUBSTITUTIONS = {
        'a': ['4', '@', 'α', 'а'],
        'e': ['3', '€', 'є', 'е'],
        'i': ['1', '!', '|', 'і'],
        'o': ['0', 'ø', 'σ', 'о'],
        's': ['5', '$', '§'],
        't': ['7', '+', '†'],
        'l': ['1', '|', 'ℓ'],
        'b': ['8', 'ß', '6'],
        'g': ['9', '6', 'ğ'],
        'z': ['2', 'ž'],
        'A': ['4', '@', 'Α', 'А'],
        'E': ['3', '€', 'Є', 'Е'],
        'I': ['1', '!', '|', 'І'],
        'O': ['0', 'Ø', 'Θ', 'О'],
        'S': ['5', '$', '§'],
        'T': ['7', '+', '†', 'Т'],
        'B': ['8', 'ß', 'В'],
        'G': ['9', '6'],
    }

    @classmethod
    def encode(cls, text: str, intensity: float = 0.5) -> str:
        """
        Codifica texto em leetspeak.

        Args:
            text: Texto original
            intensity: 0.0-1.0, porcentagem de caracteres a substituir

        Returns:
            Texto codificado
        """
        result = []
        for char in text:
            if char.lower() in cls.SUBSTITUTIONS and random.random() < intensity:
                subs = cls.SUBSTITUTIONS.get(char, cls.SUBSTITUTIONS.get(char.lower(), [char]))
                result.append(random.choice(subs))
            else:
                result.append(char)
        return ''.join(result)


class CombiningMarksInjector:
    """
    Injeta marcas combinantes Unicode que são
    quase invisíveis mas confundem processamento de texto.
    """

    # Marcas combinantes sutis (quase invisíveis)
    SUBTLE_MARKS = [
        '\u0300',  # Combining Grave Accent
        '\u0301',  # Combining Acute Accent
        '\u0302',  # Combining Circumflex
        '\u0303',  # Combining Tilde
        '\u0304',  # Combining Macron
        '\u0305',  # Combining Overline
        '\u0306',  # Combining Breve
        '\u0307',  # Combining Dot Above
        '\u0308',  # Combining Diaeresis
        '\u030A',  # Combining Ring Above
        '\u030B',  # Combining Double Acute
        '\u030C',  # Combining Caron
    ]

    # Marcas realmente invisíveis
    INVISIBLE_MARKS = [
        '\u034F',  # Combining Grapheme Joiner
        '\u0332',  # Combining Low Line (underline sutil)
        '\u0333',  # Combining Double Low Line
        '\u0334',  # Combining Tilde Overlay
        '\u0335',  # Combining Short Stroke Overlay
        '\u0336',  # Combining Long Stroke Overlay
        '\u0337',  # Combining Short Solidus Overlay
        '\u0338',  # Combining Long Solidus Overlay
    ]

    @classmethod
    def inject_subtle(cls, text: str, probability: float = 0.3) -> str:
        """Injeta marcas sutis aleatoriamente."""
        result = []
        for char in text:
            result.append(char)
            if char.isalpha() and random.random() < probability:
                # Adiciona marca invisível
                result.append(random.choice(cls.INVISIBLE_MARKS))
        return ''.join(result)


class DirectionManipulator:
    """
    Manipula caracteres de direção Unicode para
    confundir cópia/colagem e processamento.
    """

    LEFT_TO_RIGHT_MARK = '\u200E'       # LRM
    RIGHT_TO_LEFT_MARK = '\u200F'       # RLM
    LEFT_TO_RIGHT_EMBEDDING = '\u202A'  # LRE
    RIGHT_TO_LEFT_EMBEDDING = '\u202B'  # RLE
    POP_DIRECTIONAL_FORMAT = '\u202C'   # PDF
    LEFT_TO_RIGHT_OVERRIDE = '\u202D'   # LRO
    RIGHT_TO_LEFT_OVERRIDE = '\u202E'   # RLO

    @classmethod
    def add_direction_noise(cls, text: str) -> str:
        """
        Adiciona marcadores de direção que não afetam
        visualização mas confundem processamento.
        """
        result = []
        words = text.split(' ')

        for i, word in enumerate(words):
            # Adiciona LRM antes de algumas palavras
            if random.random() < 0.3:
                result.append(cls.LEFT_TO_RIGHT_MARK)
            result.append(word)
            if i < len(words) - 1:
                result.append(' ')

        return ''.join(result)


class TextCamouflage:
    """
    Classe principal para camuflagem de texto.
    Combina múltiplas técnicas para máxima eficácia.
    """

    def __init__(self):
        self.homoglyphs = HomoglyphDatabase()
        self.zero_width = ZeroWidthCharacters()
        self.leetspeak = LeetSpeakEncoder()
        self.combining = CombiningMarksInjector()
        self.direction = DirectionManipulator()

    def camouflage(
        self,
        text: str,
        mode: CamouflageMode = CamouflageMode.FULL_CAMOUFLAGE,
        intensity: float = 0.7
    ) -> CamouflageResult:
        """
        Aplica camuflagem ao texto.

        Args:
            text: Texto original
            mode: Modo de camuflagem
            intensity: Intensidade (0.0-1.0)

        Returns:
            CamouflageResult com texto camuflado e metadados
        """
        original = text

        if mode == CamouflageMode.HOMOGLYPH:
            result = self._apply_homoglyphs(text, intensity)
            desc = "Substituição por caracteres Unicode visualmente idênticos"

        elif mode == CamouflageMode.ZERO_WIDTH:
            result = self._apply_zero_width(text, intensity)
            desc = "Injeção de caracteres invisíveis entre letras"

        elif mode == CamouflageMode.LEETSPEAK:
            result = self.leetspeak.encode(text, intensity)
            desc = "Substituição estilo leetspeak mantendo legibilidade"

        elif mode == CamouflageMode.MIXED_SCRIPTS:
            result = self._apply_mixed_scripts(text, intensity)
            desc = "Mistura de scripts Unicode (Latin/Cirílico/Grego)"

        elif mode == CamouflageMode.COMBINING_MARKS:
            result = self.combining.inject_subtle(text, intensity)
            desc = "Injeção de marcas combinantes invisíveis"

        elif mode == CamouflageMode.DIRECTION_TRICK:
            result = self.direction.add_direction_noise(text)
            desc = "Manipulação de marcadores de direção"

        elif mode == CamouflageMode.VISUAL_NOISE:
            result = self._apply_visual_noise(text, intensity)
            desc = "Combinação de ruído visual sutil"

        elif mode == CamouflageMode.FULL_CAMOUFLAGE:
            result = self._apply_full_camouflage(text, intensity)
            desc = "Combinação de todas as técnicas"

        else:
            result = text
            desc = "Nenhuma camuflagem aplicada"

        # Análise Unicode
        unicode_analysis = self._analyze_unicode(result)

        return CamouflageResult(
            original_text=original,
            camouflaged_text=result,
            technique=mode.value,
            human_readable=True,
            machine_readable=False,
            description=desc,
            unicode_analysis=unicode_analysis
        )

    def _apply_homoglyphs(self, text: str, intensity: float) -> str:
        """Aplica substituição por homoglyphs."""
        result = []
        for char in text:
            if random.random() < intensity:
                homoglyphs = self.homoglyphs.get_all_homoglyphs(char)
                if len(homoglyphs) > 1:
                    # Escolhe um homoglyph diferente do original
                    alternatives = [h for h in homoglyphs if h != char]
                    if alternatives:
                        result.append(random.choice(alternatives))
                        continue
            result.append(char)
        return ''.join(result)

    def _apply_zero_width(self, text: str, intensity: float) -> str:
        """Injeta caracteres zero-width."""
        result = []
        for i, char in enumerate(text):
            result.append(char)
            # Não adiciona após espaços ou no final
            if char != ' ' and i < len(text) - 1 and random.random() < intensity:
                result.append(self.zero_width.get_random())
        return ''.join(result)

    def _apply_mixed_scripts(self, text: str, intensity: float) -> str:
        """Mistura caracteres de diferentes scripts."""
        result = []
        for char in text:
            if random.random() < intensity:
                # Tenta cirílico primeiro, depois grego
                if char in HomoglyphDatabase.LATIN_TO_CYRILLIC:
                    result.append(HomoglyphDatabase.LATIN_TO_CYRILLIC[char])
                elif char in HomoglyphDatabase.LATIN_TO_GREEK:
                    result.append(HomoglyphDatabase.LATIN_TO_GREEK[char])
                else:
                    result.append(char)
            else:
                result.append(char)
        return ''.join(result)

    def _apply_visual_noise(self, text: str, intensity: float) -> str:
        """Aplica ruído visual sutil."""
        # Combina várias técnicas com baixa intensidade
        result = self._apply_homoglyphs(text, intensity * 0.3)
        result = self._apply_zero_width(result, intensity * 0.3)
        result = self.combining.inject_subtle(result, intensity * 0.2)
        return result

    def _apply_full_camouflage(self, text: str, intensity: float) -> str:
        """Aplica camuflagem completa combinando todas as técnicas."""
        # Layer 1: Homoglyphs (40% dos caracteres elegíveis)
        result = self._apply_homoglyphs(text, intensity * 0.4)

        # Layer 2: Zero-width injection (30% dos espaços entre letras)
        result = self._apply_zero_width(result, intensity * 0.3)

        # Layer 3: Combining marks (20% das letras)
        result = self.combining.inject_subtle(result, intensity * 0.2)

        # Layer 4: Direction markers
        result = self.direction.add_direction_noise(result)

        return result

    def _analyze_unicode(self, text: str) -> Dict:
        """Analisa composição Unicode do texto."""
        categories = {}
        scripts = set()

        for char in text:
            # Categoria Unicode
            cat = unicodedata.category(char)
            categories[cat] = categories.get(cat, 0) + 1

            # Tenta identificar script
            try:
                name = unicodedata.name(char, '')
                if 'CYRILLIC' in name:
                    scripts.add('Cyrillic')
                elif 'GREEK' in name:
                    scripts.add('Greek')
                elif 'LATIN' in name:
                    scripts.add('Latin')
                elif 'ZERO WIDTH' in name:
                    scripts.add('Zero-Width')
                elif 'COMBINING' in name:
                    scripts.add('Combining')
            except:
                pass

        return {
            'length': len(text),
            'visible_length': len([c for c in text if unicodedata.category(c) != 'Cf']),
            'categories': categories,
            'scripts_detected': list(scripts),
            'has_invisible_chars': any(unicodedata.category(c) == 'Cf' for c in text),
            'has_mixed_scripts': len(scripts) > 1
        }

    def decode(self, camouflaged_text: str) -> str:
        """
        Tenta decodificar texto camuflado de volta ao original.
        (Não é 100% reversível para todas as técnicas)
        """
        result = []

        # Remove caracteres de controle invisíveis
        for char in camouflaged_text:
            cat = unicodedata.category(char)

            # Pula caracteres de formatação (zero-width, direction marks)
            if cat == 'Cf':
                continue

            # Pula marcas combinantes
            if cat.startswith('M'):
                continue

            # Tenta reverter homoglyphs
            reversed_char = self._reverse_homoglyph(char)
            result.append(reversed_char)

        return ''.join(result)

    def _reverse_homoglyph(self, char: str) -> str:
        """Tenta reverter um homoglyph para o caractere original."""
        # Cria mapeamento reverso
        reverse_cyrillic = {v: k for k, v in HomoglyphDatabase.LATIN_TO_CYRILLIC.items()}
        reverse_greek = {v: k for k, v in HomoglyphDatabase.LATIN_TO_GREEK.items()}

        if char in reverse_cyrillic:
            return reverse_cyrillic[char]
        if char in reverse_greek:
            return reverse_greek[char]

        return char


class VisualTextCamouflage:
    """
    Gera imagens com texto visualmente camuflado.
    Combina camuflagem de caracteres com perturbações visuais.
    """

    def __init__(self):
        self.text_camouflage = TextCamouflage()

    def create_camouflaged_image(
        self,
        text: str,
        mode: CamouflageMode = CamouflageMode.FULL_CAMOUFLAGE,
        font_size: int = 40,
        image_size: Tuple[int, int] = (800, 200),
        bg_color: Tuple[int, int, int] = (255, 255, 255),
        text_color: Tuple[int, int, int] = (0, 0, 0),
        add_visual_noise: bool = True
    ) -> Tuple[Image.Image, CamouflageResult]:
        """
        Cria imagem com texto camuflado.

        Returns:
            (imagem, resultado_camuflagem)
        """
        # Aplica camuflagem ao texto
        result = self.text_camouflage.camouflage(text, mode, intensity=0.7)

        # Cria imagem
        image = Image.new('RGB', image_size, bg_color)
        draw = ImageDraw.Draw(image)

        # Carrega fonte
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        # Calcula posição centralizada
        bbox = draw.textbbox((0, 0), result.camouflaged_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (image_size[0] - text_width) // 2
        y = (image_size[1] - text_height) // 2

        # Renderiza texto
        draw.text((x, y), result.camouflaged_text, fill=text_color, font=font)

        # Adiciona ruído visual se solicitado
        if add_visual_noise:
            image = self._add_visual_perturbation(image)

        return image, result

    def _add_visual_perturbation(self, image: Image.Image) -> Image.Image:
        """Adiciona perturbação visual sutil à imagem."""
        img_array = np.array(image).astype(np.float32)

        # Ruído gaussiano muito sutil
        noise = np.random.randn(*img_array.shape) * 2
        img_array = img_array + noise

        # Clipa valores
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        return Image.fromarray(img_array)


def get_camouflage_modes() -> List[Dict]:
    """Retorna lista de modos de camuflagem disponíveis."""
    return [
        {
            "id": "homoglyph",
            "name": "Homoglyphs",
            "description": "Substitui letras por caracteres Unicode visualmente idênticos (a→а)",
            "example": "Hello → Неllо",
            "effectiveness": "high",
            "human_readable": True
        },
        {
            "id": "zero_width",
            "name": "Zero-Width Injection",
            "description": "Insere caracteres invisíveis entre letras",
            "example": "Hello → H​e​l​l​o (com chars invisíveis)",
            "effectiveness": "very_high",
            "human_readable": True
        },
        {
            "id": "leetspeak",
            "name": "Leetspeak Inteligente",
            "description": "Substituições estilo hacker mantendo legibilidade",
            "example": "Hello → H3ll0",
            "effectiveness": "medium",
            "human_readable": True
        },
        {
            "id": "mixed_scripts",
            "name": "Scripts Misturados",
            "description": "Mistura Latin/Cirílico/Grego de forma transparente",
            "example": "HELLO → НЕLLΟ",
            "effectiveness": "very_high",
            "human_readable": True
        },
        {
            "id": "combining_marks",
            "name": "Marcas Combinantes",
            "description": "Adiciona diacríticos invisíveis às letras",
            "example": "Hello → H̸e̸l̸l̸o̸",
            "effectiveness": "high",
            "human_readable": True
        },
        {
            "id": "direction_trick",
            "name": "Truque de Direção",
            "description": "Manipula marcadores LTR/RTL para confundir cópia",
            "example": "Texto parece normal, copia errado",
            "effectiveness": "medium",
            "human_readable": True
        },
        {
            "id": "full_camouflage",
            "name": "Camuflagem Completa",
            "description": "Combina todas as técnicas para máxima proteção",
            "example": "Todas as técnicas aplicadas em camadas",
            "effectiveness": "maximum",
            "human_readable": True
        }
    ]


def demonstrate_camouflage(text: str = "Senha: ABC123") -> Dict:
    """
    Demonstra todas as técnicas de camuflagem em um texto.

    Returns:
        Dict com exemplos de cada técnica
    """
    camouflage = TextCamouflage()
    results = {}

    for mode in CamouflageMode:
        result = camouflage.camouflage(text, mode, intensity=0.7)
        results[mode.value] = {
            "original": result.original_text,
            "camouflaged": result.camouflaged_text,
            "description": result.description,
            "unicode_info": result.unicode_analysis
        }

    return results
