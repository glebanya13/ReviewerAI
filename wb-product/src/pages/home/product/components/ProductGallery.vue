<template>
  <div class="product-gallery">
    <div class="main-image-container">
      <img 
        :src="currentImage" 
        :alt="`Изображение товара ${currentIndex + 1}`"
        class="main-image"
        @click="openLightbox"
      />
      
      <button 
        v-if="images.length > 1"
        class="nav-arrow nav-arrow-left" 
        @click="previousImage"
        :disabled="currentIndex === 0"
      >
        ‹
      </button>
      <button 
        v-if="images.length > 1"
        class="nav-arrow nav-arrow-right" 
        @click="nextImage"
        :disabled="currentIndex === images.length - 1"
      >
        ›
      </button>
    </div>

    <div v-if="images.length > 1" class="thumbnails">
      <div 
        v-for="(image, index) in images" 
        :key="index"
        class="thumbnail"
        :class="{ active: index === currentIndex }"
        @click="setCurrentImage(index)"
      >
        <img :src="image" :alt="`Миниатюра ${index + 1}`" />
      </div>
    </div>

    <div v-if="lightboxOpen" class="lightbox" @click="closeLightbox">
      <div class="lightbox-content" @click.stop>
        <button class="lightbox-close" @click="closeLightbox">×</button>
        <img :src="currentImage" :alt="`Увеличенное изображение ${currentIndex + 1}`" />
        <button 
          v-if="images.length > 1"
          class="lightbox-nav lightbox-nav-left" 
          @click="previousImage"
        >
          ‹
        </button>
        <button 
          v-if="images.length > 1"
          class="lightbox-nav lightbox-nav-right" 
          @click="nextImage"
        >
          ›
        </button>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ProductGallery',
  props: {
    images: {
      type: Array,
      required: true,
      default: () => []
    }
  },
  data() {
    return {
      currentIndex: 0,
      lightboxOpen: false
    }
  },
  computed: {
    currentImage() {
      return this.images[this.currentIndex] || this.images[0] || ''
    }
  },
  methods: {
    setCurrentImage(index) {
      this.currentIndex = index
    },
    nextImage() {
      if (this.currentIndex < this.images.length - 1) {
        this.currentIndex++
      }
    },
    previousImage() {
      if (this.currentIndex > 0) {
        this.currentIndex--
      }
    },
    openLightbox() {
      this.lightboxOpen = true
      document.body.style.overflow = 'hidden'
    },
    closeLightbox() {
      this.lightboxOpen = false
      document.body.style.overflow = 'auto'
    }
  },
  mounted() {
    const handleKeydown = (e) => {
      if (!this.lightboxOpen) return
      
      switch (e.key) {
        case 'Escape':
          this.closeLightbox()
          break
        case 'ArrowLeft':
          this.previousImage()
          break
        case 'ArrowRight':
          this.nextImage()
          break
      }
    }
    
    document.addEventListener('keydown', handleKeydown)
    
    this.keydownHandler = handleKeydown
  },
  beforeUnmount() {
    if (this.keydownHandler) {
      document.removeEventListener('keydown', this.keydownHandler)
    }
  }
}
</script>

<style scoped>
.product-gallery {
  position: relative;
}

.main-image-container {
  position: relative;
  margin-bottom: 12px;
  border-radius: 8px;
  overflow: hidden;
  background: var(--mo-neutral-950);
}

.main-image {
  width: 100%;
  height: 500px;
  object-fit: cover;
  cursor: zoom-in;
  transition: transform 0.2s ease;
  border-radius: 8px;
}

.main-image:hover {
  transform: scale(1.02);
}

.nav-arrow {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  background: var(--mo-neutral-1000);
  color: var(--mo-neutral-300);
  border: none;
  width: 32px;
  height: 32px;
  border-radius: 50%;
  font-size: 16px;
  cursor: pointer;
  transition: all 0.2s ease;
  z-index: 10;
  box-shadow: 0 2px 8px var(--mo-neutral-0A20);
}

.nav-arrow:hover {
  background: var(--mo-neutral-950);
  box-shadow: 0 4px 12px var(--mo-neutral-0A30);
}

.nav-arrow:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.nav-arrow-left {
  left: 12px;
}

.nav-arrow-right {
  right: 12px;
}

.thumbnails {
  display: flex;
  gap: 6px;
  overflow-x: auto;
  padding: 6px 0;
}

.thumbnail {
  flex-shrink: 0;
  width: 64px;
  height: 64px;
  border-radius: 6px;
  overflow: hidden;
  cursor: pointer;
  border: 2px solid transparent;
  transition: all 0.2s ease;
  background: var(--mo-neutral-950);
}

.thumbnail:hover {
  border-color: #cb11ab;
}

.thumbnail.active {
  border-color: #cb11ab;
  box-shadow: 0 0 0 1px rgba(203, 17, 171, 0.2);
}

.thumbnail img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

/* Lightbox */
.lightbox {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.9);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  cursor: pointer;
}

.lightbox-content {
  position: relative;
  max-width: 90vw;
  max-height: 90vh;
  cursor: default;
}

.lightbox-content img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  border-radius: 8px;
}

.lightbox-close {
  position: absolute;
  top: -40px;
  right: 0;
  background: none;
  border: none;
  color: white;
  font-size: 32px;
  cursor: pointer;
  z-index: 1001;
}

.lightbox-nav {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  background: rgba(255, 255, 255, 0.2);
  color: white;
  border: none;
  width: 50px;
  height: 50px;
  border-radius: 50%;
  font-size: 24px;
  cursor: pointer;
  transition: all 0.2s ease;
  z-index: 1001;
}

.lightbox-nav:hover {
  background: rgba(255, 255, 255, 0.3);
}

.lightbox-nav-left {
  left: -60px;
}

.lightbox-nav-right {
  right: -60px;
}

/* Responsive */
@media (max-width: 768px) {
  .main-image {
    height: 300px;
  }
  
  .thumbnail {
    width: 60px;
    height: 60px;
  }
  
  .nav-arrow {
    width: 32px;
    height: 32px;
    font-size: 16px;
  }
  
  .lightbox-nav {
    width: 40px;
    height: 40px;
    font-size: 20px;
  }
  
  .lightbox-nav-left {
    left: -50px;
  }
  
  .lightbox-nav-right {
    right: -50px;
  }
}
</style>
