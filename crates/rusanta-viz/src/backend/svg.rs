use std::fmt::Write;

use crate::backend::Backend;
use crate::style::Color;

/// SVG backend.
///
/// Renders plots into an SVG string buffer.
pub struct SvgBackend {
    width: u32,
    height: u32,
    buffer: String,
}

impl SvgBackend {
    /// Create a new SVG backend.
    pub fn new(width: u32, height: u32) -> Self {
        let mut buffer = String::new();
        writeln!(
            buffer,
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">"#,
            w = width,
            h = height
        )
        .unwrap();

        Self {
            width,
            height,
            buffer,
        }
    }

    /// Finish rendering and return SVG string.
    pub fn finalize(mut self) -> String {
        self.buffer.push_str("</svg>\n");
        self.buffer
    }

    fn color_to_svg(color: Color) -> String {
        format!(
            "rgba({},{},{},{})",
                (color.r * 255.0) as u8,
                (color.g * 255.0) as u8,
                (color.b * 255.0) as u8,
                color.a
        )
    }

    fn map_x(&self, x: f64) -> f64 {
        x * self.width as f64
    }

    fn map_y(&self, y: f64) -> f64 {
        self.height as f64 - y * self.height as f64
    }
}

impl Backend for SvgBackend {
    fn draw_line(
        &mut self,
        x: &[f64],
        y: &[f64],
        width: f32,
        color: Color,
        label: Option<&str>,
    ) -> crate::Result<()> {
        let mut points = String::new();
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let _ = write!(
                points,
                "{},{} ",
                self.map_x(xi),
                           self.map_y(yi)
            );
        }

        writeln!(
            self.buffer,
            r#"<polyline points="{}" fill="none" stroke="{}" stroke-width="{}"/>"#,
            points,
            Self::color_to_svg(color),
                 width
        )
        .unwrap();

        if let Some(label) = label {
            let _ = label; // legend support comes later
        }

        Ok(())
    }

    fn draw_scatter(
        &mut self,
        x: &[f64],
        y: &[f64],
        size: f32,
        color: Color,
        label: Option<&str>,
    ) -> crate::Result<()> {
        let radius = size / 2.0;

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            writeln!(
                self.buffer,
                r#"<circle cx="{}" cy="{}" r="{}" fill="{}"/>"#,
                self.map_x(xi),
                     self.map_y(yi),
                     radius,
                     Self::color_to_svg(color)
            )
            .unwrap();
        }

        if let Some(label) = label {
            let _ = label;
        }

        Ok(())
    }

    fn draw_bars(
        &mut self,
        x: &[f64],
        heights: &[f64],
        width: f64,
        color: Color,
        label: Option<&str>,
    ) -> crate::Result<()> {
        for (&xi, &hi) in x.iter().zip(heights.iter()) {
            let bar_width = width * self.width as f64;
            let x0 = self.map_x(xi) - bar_width / 2.0;
            let y0 = self.map_y(hi);
            let h = hi * self.height as f64;

            writeln!(
                self.buffer,
                r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}"/>"#,
                x0,
                y0,
                bar_width,
                h,
                Self::color_to_svg(color)
            )
            .unwrap();
        }

        if let Some(label) = label {
            let _ = label;
        }

        Ok(())
    }
}
