from pathlib import Path

def draw(path):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    cells=[]
    for row,stride in enumerate([32,33]):
        y=100+row*110
        cells.append(f'<text x="20" y="{y+20}" font-size="16">stride={stride}</text>')
        for lane in range(32):
            bank=(lane*stride)%32;x=140+lane*37
            fill=f'hsl({bank*360/32:.1f},65%,78%)'
            cells.append(f'<rect x="{x}" y="{y}" width="34" height="40" fill="{fill}" stroke="#555"/>')
            cells.append(f'<text x="{x+17}" y="{y+25}" text-anchor="middle" font-size="13">{bank}</text>')
            if row==0:cells.append(f'<text x="{x+17}" y="{y-12}" text-anchor="middle" font-size="12">{lane}</text>')
        desc='32 lanes read different words in bank 0 (32-way conflict)' if stride==32 else '32 lanes read banks 0..31 (no bank collision for this load)'
        cells.append(f'<text x="140" y="{y+64}" font-size="15">{desc}</text>')
    svg='<svg xmlns="http://www.w3.org/2000/svg" width="1360" height="340" viewBox="0 0 1360 340"><rect width="1360" height="340" fill="white"/><g font-family="monospace" fill="#222"><text x="20" y="30" font-size="19">Transpose column load: tile[lane][0], float32, 32 banks</text><text x="20" y="55" font-size="15">bank = (lane * row_stride + column) % 32; column = 0. Top labels: lane. Boxes: bank.</text>'+''.join(cells)+'</g></svg>'
    path.write_text(svg)
if __name__=='__main__':draw(Path(__file__).resolve().parent/'results/bank_mapping.svg')
