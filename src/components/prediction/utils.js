export function fmtPrice(v) {
    if (v == null || isNaN(v)) return "—";
    if (v >= 1_000_000) return "$" + (v / 1_000_000).toFixed(2) + "M";
    if (v >= 1_000)     return "$" + v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    if (v >= 1)         return "$" + v.toFixed(4);
    return "$" + v.toFixed(6);
}
export function fmtPct(v) {
    if (v == null || isNaN(v)) return "—";
    return (v >= 0 ? "+" : "") + v.toFixed(2) + "%";
}
export function calcChange(current, forecast) {
    if (!current || !forecast) return null;
    return ((forecast - current) / current) * 100;
}
export function fmtCompact(v) {
    if (v == null || isNaN(v)) return "—";
    if (Math.abs(v) >= 1e9) return "$" + (v / 1e9).toFixed(2) + "B";
    if (Math.abs(v) >= 1e6) return "$" + (v / 1e6).toFixed(2) + "M";
    if (Math.abs(v) >= 1e3) return "$" + (v / 1e3).toFixed(1) + "K";
    return "$" + v.toFixed(2);
}
