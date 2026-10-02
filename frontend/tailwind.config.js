/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        base: {
          900: "#0a0e17",
          800: "#0f1523",
          700: "#161d2e",
          600: "#1e2740",
        },
        line: "#243049",
        accent: {
          blue: "#4f8cff",
          purple: "#8b7dff",
          cyan: "#38d9c6",
        },
        muted: "#8a97b1",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
