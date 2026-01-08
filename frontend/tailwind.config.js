/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'lab-primary': '#3B82F6',
        'lab-secondary': '#10B981',
        'lab-danger': '#EF4444',
        'lab-dark': '#1F2937',
        'lab-darker': '#111827',
      }
    },
  },
  plugins: [],
}
