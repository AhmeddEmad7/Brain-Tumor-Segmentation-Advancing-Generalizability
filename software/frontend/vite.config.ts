import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
    plugins: [
        react(),
        {
          name: 'cross-origin-isolation-headers',
          configureServer(server) {
            // server.middlewares.use((req, res, next) => {
            //   // Only apply headers if the host header includes "localhost:707"
            //   if (req.headers.host && req.headers.host.includes("localhost:7070")) {
            //     res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
            //     res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
            //   }
            //   next();
            // });
          }
        }
      ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src/'),
      '@assets': path.resolve(__dirname, './src/assets/'),
      '@context': path.resolve(__dirname, './src/context/'),
      '@data': path.resolve(__dirname, './src/data/'),
      '@features': path.resolve(__dirname, './src/features/'),
      '@hooks': path.resolve(__dirname, './src/hooks/'),
      '@models': path.resolve(__dirname, './src/models/'),
      '@pages': path.resolve(__dirname, './src/pages/'),
      '@services': path.resolve(__dirname, './src/services/'),
      '@styles': path.resolve(__dirname, './src/styles/'),
      '@ui': path.resolve(__dirname, './src/ui/'),
      '@utilities': path.resolve(__dirname, './src/utilities/')
    }
  },
  server: {
    host: true,
    port: 5000,
    strictPort: false
  }
});
