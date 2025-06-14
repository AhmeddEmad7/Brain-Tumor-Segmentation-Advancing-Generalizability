FROM node:18-alpine

# Install dependencies for esbuild
RUN apk add --no-cache libc6-compat

WORKDIR /app

COPY package.json ./

RUN npm install

COPY . .

EXPOSE 5000

CMD ["npm", "run", "dev"]
