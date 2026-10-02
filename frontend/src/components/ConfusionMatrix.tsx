import type { ConfusionEntry } from "../types";

// Heat-coloured confusion matrix. Colour intensity scales with the row-
// normalised value so class imbalance doesn't wash out the diagonal.
export default function ConfusionMatrix({ data }: { data: ConfusionEntry }) {
  const { matrix, labels } = data;
  const rowSums = matrix.map((row) => row.reduce((a, b) => a + b, 0) || 1);

  return (
    <div className="overflow-x-auto">
      <table className="border-collapse text-sm">
        <thead>
          <tr>
            <th className="p-2 text-xs text-muted">true \ pred</th>
            {labels.map((l) => (
              <th key={l} className="p-2 text-xs font-mono text-muted">
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <td className="p-2 text-xs font-mono text-muted">{labels[i]}</td>
              {row.map((v, j) => {
                const intensity = v / rowSums[i];
                const isDiag = i === j;
                const bg = isDiag
                  ? `rgba(56,217,198,${0.12 + intensity * 0.6})`
                  : `rgba(79,140,255,${0.06 + intensity * 0.5})`;
                return (
                  <td
                    key={j}
                    className="p-3 text-center font-mono tabular-nums"
                    style={{ background: bg, minWidth: 56 }}
                    title={`true=${labels[i]} pred=${labels[j]}: ${v}`}
                  >
                    {v}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
