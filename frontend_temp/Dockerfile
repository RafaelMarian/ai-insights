# Imagine de bază Node
FROM node:18

# Setăm directorul de lucru
WORKDIR /app

# Copiem package.json și package-lock.json
COPY package*.json ./

# Instalăm dependențele
RUN npm install

# Copiem restul aplicației
COPY . .

# Construim aplicația React
RUN npm run build

# Folosim un server web pentru a servi build-ul
FROM nginx:alpine
COPY --from=0 /app/build /usr/share/nginx/html

# Expunem portul 80
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
