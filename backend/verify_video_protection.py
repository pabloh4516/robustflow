"""
Script para verificar se a proteção de vídeo está funcionando.
Compara vídeo original vs processado.
"""

import sys
import hashlib
import subprocess
import json
from pathlib import Path

def get_file_hash(filepath):
    """Calcula hash MD5 do arquivo"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_video_info(filepath):
    """Extrai informações do vídeo usando ffprobe"""
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', '-show_streams', str(filepath)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return json.loads(result.stdout)
    except Exception as e:
        return {"error": str(e)}

def extract_frame(filepath, output_path, time="00:00:01"):
    """Extrai um frame do vídeo"""
    cmd = [
        'ffmpeg', '-y', '-i', str(filepath),
        '-ss', time, '-vframes', '1', str(output_path)
    ]
    subprocess.run(cmd, capture_output=True)

def compare_videos(original_path, protected_path):
    """Compara vídeo original com protegido"""
    print("=" * 60)
    print("VERIFICAÇÃO DE PROTEÇÃO DE VÍDEO")
    print("=" * 60)

    original = Path(original_path)
    protected = Path(protected_path)

    if not original.exists():
        print(f"ERRO: Arquivo original não encontrado: {original}")
        return
    if not protected.exists():
        print(f"ERRO: Arquivo protegido não encontrado: {protected}")
        return

    # 1. Comparar hashes
    print("\n[1] COMPARAÇÃO DE HASH")
    hash_orig = get_file_hash(original)
    hash_prot = get_file_hash(protected)

    print(f"   Original:  {hash_orig}")
    print(f"   Protegido: {hash_prot}")

    if hash_orig != hash_prot:
        print("   [OK] HASHES DIFERENTES - Arquivo foi modificado")
    else:
        print("   [FALHA] HASHES IGUAIS - Arquivo NAO foi modificado!")

    # 2. Comparar tamanhos
    print("\n[2] COMPARAÇÃO DE TAMANHO")
    size_orig = original.stat().st_size
    size_prot = protected.stat().st_size

    print(f"   Original:  {size_orig:,} bytes ({size_orig/1024/1024:.2f} MB)")
    print(f"   Protegido: {size_prot:,} bytes ({size_prot/1024/1024:.2f} MB)")
    print(f"   Diferença: {abs(size_orig - size_prot):,} bytes")

    # 3. Comparar metadados
    print("\n[3] COMPARAÇÃO DE METADADOS")
    info_orig = get_video_info(original)
    info_prot = get_video_info(protected)

    if "error" not in info_orig and "error" not in info_prot:
        # Verificar se metadados foram removidos
        orig_tags = info_orig.get('format', {}).get('tags', {})
        prot_tags = info_prot.get('format', {}).get('tags', {})

        print(f"   Metadados original: {len(orig_tags)} campos")
        print(f"   Metadados protegido: {len(prot_tags)} campos")

        if len(prot_tags) < len(orig_tags):
            print("   [OK] METADADOS REDUZIDOS - Proteção aplicada")

        # Verificar resolução
        for stream in info_orig.get('streams', []):
            if stream.get('codec_type') == 'video':
                orig_res = f"{stream.get('width')}x{stream.get('height')}"
                break

        for stream in info_prot.get('streams', []):
            if stream.get('codec_type') == 'video':
                prot_res = f"{stream.get('width')}x{stream.get('height')}"
                break

        print(f"\n   Resolução original:  {orig_res}")
        print(f"   Resolução protegida: {prot_res}")

        if orig_res != prot_res:
            print("   [OK] RESOLUÇÃO ALTERADA - Dificulta correspondência")
    else:
        print("   (ffprobe não disponível - instale ffmpeg para análise completa)")

    # 4. Extrair frames para comparação visual
    print("\n[4] EXTRAÇÃO DE FRAMES PARA COMPARAÇÃO")
    frame_orig = original.parent / "frame_original.png"
    frame_prot = protected.parent / "frame_protected.png"

    extract_frame(original, frame_orig)
    extract_frame(protected, frame_prot)

    if frame_orig.exists() and frame_prot.exists():
        print(f"   Frame original:  {frame_orig}")
        print(f"   Frame protegido: {frame_prot}")
        print("   [OK] Frames extraídos - Compare visualmente")
    else:
        print("   (ffmpeg não disponível para extração de frames)")

    # Resumo
    print("\n" + "=" * 60)
    print("RESUMO DA PROTEÇÃO")
    print("=" * 60)

    protection_score = 0
    checks = []

    if hash_orig != hash_prot:
        protection_score += 1
        checks.append("[OK] Hash alterado")
    else:
        checks.append("[FALHA] Hash igual")

    if size_orig != size_prot:
        protection_score += 1
        checks.append("[OK] Tamanho diferente")
    else:
        checks.append("[FALHA] Tamanho igual")

    for check in checks:
        print(f"   {check}")

    print(f"\n   Score de proteção: {protection_score}/2")

    if protection_score >= 2:
        print("   STATUS: [OK] VÍDEO PROTEGIDO COM SUCESSO")
    elif protection_score == 1:
        print("   STATUS: ~ PROTEÇÃO PARCIAL")
    else:
        print("   STATUS: [FALHA] PROTEÇÃO NÃO APLICADA")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python verify_video_protection.py <video_original> <video_protegido>")
        print("Exemplo: python verify_video_protection.py original.mp4 protected.mp4")
        sys.exit(1)

    compare_videos(sys.argv[1], sys.argv[2])
