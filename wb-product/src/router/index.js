import { createRouter, createWebHistory } from 'vue-router'
import ProductPage from '../pages/home/product/index.vue'
import ReviewsPage from '../pages/home/reviews/index.vue'

const routes = [
  {
    path: '/',
    name: 'Product',
    component: ProductPage,
    meta: { 
      key: 'product',
      keepAlive: false 
    }
  },
  {
    path: '/reviews',
    name: 'Reviews',
    component: ReviewsPage,
    meta: { 
      key: 'reviews',
      keepAlive: false 
    }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior(to, from, savedPosition) {
    return { 
      top: 0,
      behavior: 'smooth'
    }
  }
})

export default router
