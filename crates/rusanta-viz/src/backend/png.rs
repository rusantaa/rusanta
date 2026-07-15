use crate::backend::Backend;
use crate::style::Color;

use std::path::Path;

/// Simple software raster PNG backend.
pub struct PngBackend {
    width: u32,
    height: u32,
    buffer: Vec<u8>, // RGBA
}

impl PngBackend {
    pub fn new(width: u32, height: u32) -> Self {
        let buffer = vec![255; (width * height * 4) as usize];
        Self {
            width,
            height,
            buffer,
        }
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> crate::Result<()> {
        image::save_buffer(
            path,
            &self.buffer,
            self.width,
            self.height,
            image::ColorType::Rgba8,
        )?;
        Ok(())
    }

    fn map_x(&self, x: f64) -> i32 {
        (x * self.width as f64) as i32
    }

    fn map_y(&self, y: f64) -> i32 {
        ((1.0 - y) * self.height as f64) as i32
    }

    fn set_pixel(&mut self, x: i32, y: i32, color: Color) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return;
        }
        let idx = ((y as u32 * self.width + x as u32) * 4) as usize;
        self.buffer[idx] = (color.r * 255.0) as u8;
        self.buffer[idx + 1] = (color.g * 255.0) as u8;
        self.buffer[idx + 2] = (color.b * 255.0) as u8;
        self.buffer[idx + 3] = (color.a * 255.0) as u8;
    }

    fn draw_rect(&mut self, x: i32, y: i32, w: i32, h: i32, color: Color) {
        for yy in y..(y + h) {
            for xx in x..(x + w) {
                self.set_pixel(xx, yy, color);
            }
        }
    }
}

impl Backend for PngBackend {
    fn draw_line(
        &mut self,
        x: &[f64],
        y: &[f64],
        _width: f32,
        color: Color,
        _label: Option<&str>,
    ) -> crate::Result<()> {
        for i in 0..x.len().saturating_sub(1) {
            let x0 = self.map_x(x[i]);
            let y0 = self.map_y(y[i]);
            let x1 = self.map_x(x[i + 1]);
            let y1 = self.map_y(y[i + 1]);

            let dx = (x1 - x0).abs();
            let dy = -(y1 - y0).abs();
            let sx = if x0 < x1 { 1 } else { -1 };
            let sy = if y0 < y1 { 1 } else { -1 };
            let mut err = dx + dy;

            let mut cx = x0;
            let mut cy = y0;
            loop {
                self.set_pixel(cx, cy, color);
                if cx == x1 && cy == y1 {
                    break;
                }
                let e2 = 2 * err;
                if e2 >= dy {
                    err += dy;
                    cx += sx;
                }
                if e2 <= dx {
                    err += dx;
                    cy += sy;
                }
            }
        }
        Ok(())
    }

    fn draw_scatter(
        &mut self,
        x: &[f64],
        y: &[f64],
        size: f32,
        color: Color,
        _label: Option<&str>,
    ) -> crate::Result<()> {
        let r = (size / 2.0) as i32;
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let cx = self.map_x(xi);
            let cy = self.map_y(yi);
            for dy in -r..=r {
                for dx in -r..=r {
                    if dx * dx + dy * dy <= r * r {
                        self.set_pixel(cx + dx, cy + dy, color);
                    }
                }
            }
        }
        Ok(())
    }

    fn draw_bars(
        &mut self,
        x: &[f64],
        heights: &[f64],
        width: f64,
        color: Color,
        _label: Option<&str>,
    ) -> crate::Result<()> {
        let bar_w = (width * self.width as f64) as i32;
        for (&xi, &hi) in x.iter().zip(heights.iter()) {
            let cx = self.map_x(xi);
            let h = (hi * self.height as f64) as i32;
            let y = self.height as i32 - h;
            self.draw_rect(cx - bar_w / 2, y, bar_w, h, color);
        }
        Ok(())
    }
}
