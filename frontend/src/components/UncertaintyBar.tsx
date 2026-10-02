// Continuous uncertainty bar: fills proportionally to the uncertainty value
// (0 = certain, higher = less certain) with a colour gradient.
export default function UncertaintyBar({
  value,
  max = 1,
}: {
  value: number;
  max?: number;
}) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  const color =
    value < 0.1 ? "#38d9c6" : value < 0.25 ? "#4f8cff" : value < 0.4 ? "#f0a35e" : "#f06e6e";
  return (
    <div className="w-full">
      <div className="h-3 w-full overflow-hidden rounded-full bg-base-700">
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <div className="mt-1 flex justify-between font-mono text-[11px] text-muted">
        <span>certain</span>
        <span>{value.toFixed(3)}</span>
        <span>uncertain</span>
      </div>
    </div>
  );
}
