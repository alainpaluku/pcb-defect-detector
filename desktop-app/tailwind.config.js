/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Nunito', 'system-ui', 'sans-serif'],
      },
      colors: {
        dark: {
          900: '#0d0d0d',
          800: '#1a1a1a',
          700: '#242424',
          600: '#2d2d2d',
          500: '#3d3d3d',
        },
        lime: {
          400: '#c8f547',
          500: '#b8e636',
        },
        accent: {
          purple: '#8b5cf6',
          blue: '#3b82f6',
          orange: '#f97316',
          pink: '#ec4899',
        }
      },
      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.5rem',
      }
    },
  },
  plugins: [],
}
