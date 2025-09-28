<template>
  <div id="app">
    <header class="header">
      <div class="container">
        <div class="header-content">
          <router-link to="/" class="logo">Wildberries</router-link>
          <nav>
            <router-link to="/" class="btn btn-secondary">Каталог</router-link>
          </nav>
        </div>
      </div>
    </header>

    <main class="product-page">
      <div class="container">
        <router-view :key="$route.path" />
      </div>
    </main>
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>

<style>
#app {
  min-height: 100vh;
}
</style>
