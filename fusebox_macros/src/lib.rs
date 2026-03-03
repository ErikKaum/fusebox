use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use syn::{
    Attribute, Data, DeriveInput, Error, Fields, LitBool, LitStr, PathArguments, Token, Type,
    parse_macro_input, spanned::Spanned,
};

#[proc_macro_derive(Module, attributes(param, module))]
pub fn derive_module(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand_module(&input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn expand_module(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let struct_name = &input.ident;

    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(n) => &n.named,
            _ => {
                return Err(Error::new(
                    input.span(),
                    "#[derive(Module)] requires a struct with named fields",
                ));
            }
        },
        _ => {
            return Err(Error::new(
                input.span(),
                "#[derive(Module)] only supports structs",
            ));
        }
    };

    let mut init_stmts = Vec::new();
    let mut field_inits = Vec::new();

    for field in fields {
        let ident = field
            .ident
            .as_ref()
            .ok_or_else(|| Error::new(field.span(), "expected named field"))?;

        let field_name = ident.to_string();

        let param_opts = param_opts(&field.attrs)?;
        let module_opts = module_opts(&field.attrs)?;

        let kind = classify_field_type(&field.ty)?;

        match kind {
            FieldKind::Tensor | FieldKind::OptionTensor => {
                if param_opts.name.is_none() && module_opts.name.is_some() {
                    return Err(Error::new(
                        field.span(),
                        r#"use #[param(name = "...")] to rename Tensor fields, not #[module(name = "...")]"#,
                    ));
                }
                if module_opts.skip {
                    return Err(Error::new(
                        field.span(),
                        r#"#[module(skip)] is for non-Tensor config/submodules, not Tensor fields"#,
                    ));
                }

                let key_name = param_opts.name.unwrap_or_else(|| field_name.clone());
                let key_lit = LitStr::new(&key_name, ident.span());
                let local_lit = LitStr::new(&key_name, ident.span());

                match kind {
                    FieldKind::Tensor => {
                        init_stmts.push(quote! {
                            let __key = cx.qualify(#key_lit);
                            let __shape = shapes.shape_of(&__key)?
                                .ok_or(::fusebox::error::Error::MissingWeight { key: __key })?;
                            let #ident = cx.weight(#local_lit, __shape);
                        });
                    }
                    FieldKind::OptionTensor => {
                        init_stmts.push(quote! {
                            let __key = cx.qualify(#key_lit);
                            let #ident = match shapes.shape_of(&__key)? {
                                Some(__shape) => Some(cx.weight(#local_lit, __shape)),
                                None => None,
                            };
                        });
                    }
                    _ => unreachable!(),
                }
            }

            FieldKind::VecModule(ref inner_ty) => {
                if param_opts.name.is_some() {
                    return Err(Error::new(
                        field.span(),
                        r#"use #[module(...)] on Vec<T> fields; #[param(...)] is only for Tensor fields"#,
                    ));
                }
                if module_opts.skip {
                    init_stmts.push(quote! {
                        let #ident: Vec<#inner_ty> = Vec::new();
                    });
                } else {
                    let sub_name = module_opts.name.unwrap_or_else(|| field_name.clone());
                    let sub_lit = LitStr::new(&sub_name, ident.span());

                    init_stmts.push(quote! {
                        let #ident = {
                            let __scope = cx.push_scope(#sub_lit);
                            let mut __items: Vec<#inner_ty> = Vec::new();
                            let mut __idx: usize = 0;
                            loop {
                                let __probe = format!("{}/", cx.qualify(&__idx.to_string()));
                                if !shapes.has_prefix(&__probe) { break; }
                                __items.push(
                                    <#inner_ty as ::fusebox::module_api::Module>::trace(
                                        cx,
                                        &__idx.to_string(),
                                        shapes,
                                    )?
                                );
                                __idx += 1;
                            }
                            cx.pop_scope(__scope);
                            __items
                        };
                    });
                }
            }

            FieldKind::SubModule => {
                if param_opts.name.is_some() {
                    return Err(Error::new(
                        field.span(),
                        r#"use #[module(...)] on submodules/config fields; #[param(...)] is only for Tensor fields"#,
                    ));
                }

                let ty = &field.ty;

                if module_opts.skip {
                    init_stmts.push(quote! {
                        let #ident: #ty = ::core::default::Default::default();
                    });
                } else {
                    if looks_like_config_type(&field.ty) {
                        let ty_s = field.ty.to_token_stream().to_string();
                        return Err(Error::new(
                            field.ty.span(),
                            format!(
                                "field `{}` has type `{}` which is not Tensor/Option<Tensor>; \
                                 the derive macro would treat it as a submodule and try to call `Module::trace` on it. \
                                 Add #[module(skip)] to mark it as config (or change the type).",
                                field_name, ty_s
                            ),
                        ));
                    }

                    let sub_name = module_opts.name.unwrap_or_else(|| field_name.clone());
                    let sub_lit = LitStr::new(&sub_name, ident.span());

                    init_stmts.push(quote! {
                        let #ident = <#ty as ::fusebox::module_api::Module>::trace(cx, #sub_lit, shapes)?;
                    });
                }
            }
        }

        field_inits.push(quote! { #ident });
    }

    Ok(quote! {
        impl ::fusebox::module_api::Module for #struct_name {
            fn trace(
                cx: &mut ::fusebox::trace::TraceCx,
                name: &str,
                shapes: &dyn ::fusebox::module_api::ShapeProvider,
            ) -> Result<Self, ::fusebox::error::Error> {
                let __scope = cx.push_scope(name);

                #(#init_stmts)*

                cx.pop_scope(__scope);
                Ok(Self { #(#field_inits),* })
            }
        }
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum FieldKind {
    Tensor,
    OptionTensor,
    VecModule(Type),
    SubModule,
}

fn classify_field_type(ty: &Type) -> syn::Result<FieldKind> {
    match ty {
        Type::Path(tp) => {
            let seg = tp
                .path
                .segments
                .last()
                .ok_or_else(|| Error::new(tp.span(), "empty type path"))?;

            if seg.ident == "Tensor" {
                return Ok(FieldKind::Tensor);
            }

            if seg.ident == "Option" {
                let args = match &seg.arguments {
                    PathArguments::AngleBracketed(ab) => ab,
                    _ => return Ok(FieldKind::SubModule),
                };

                if args.args.len() != 1 {
                    return Ok(FieldKind::SubModule);
                }

                let inner = args.args.first().unwrap();
                if let syn::GenericArgument::Type(Type::Path(inner_tp)) = inner {
                    let inner_seg = inner_tp.path.segments.last();
                    let is_tensor = inner_seg.map(|s| s.ident == "Tensor").unwrap_or(false);
                    return Ok(if is_tensor {
                        FieldKind::OptionTensor
                    } else {
                        FieldKind::SubModule
                    });
                }

                return Ok(FieldKind::SubModule);
            }

            if seg.ident == "Vec" {
                let args = match &seg.arguments {
                    PathArguments::AngleBracketed(ab) => ab,
                    _ => return Ok(FieldKind::SubModule),
                };
                if args.args.len() == 1 {
                    if let syn::GenericArgument::Type(inner_ty) = args.args.first().unwrap() {
                        if let Type::Path(inner_tp) = inner_ty {
                            let is_tensor = inner_tp
                                .path
                                .segments
                                .last()
                                .map(|s| s.ident == "Tensor")
                                .unwrap_or(false);
                            if !is_tensor {
                                return Ok(FieldKind::VecModule(inner_ty.clone()));
                            }
                        }
                    }
                }
                return Ok(FieldKind::SubModule);
            }

            Ok(FieldKind::SubModule)
        }
        _ => Ok(FieldKind::SubModule),
    }
}

#[derive(Default)]
struct ParamOpts {
    name: Option<String>,
}

fn param_opts(attrs: &[Attribute]) -> syn::Result<ParamOpts> {
    let mut out = ParamOpts::default();

    for a in attrs {
        if !a.path().is_ident("param") {
            continue;
        }

        if out.name.is_some() {
            return Err(Error::new(a.span(), "duplicate #[param(...)]"));
        }

        a.parse_nested_meta(|meta| {
            if meta.path.is_ident("name") {
                let value: LitStr = meta.value()?.parse()?;
                out.name = Some(value.value());
                Ok(())
            } else {
                Err(meta.error("expected `name`"))
            }
        })?;
    }

    Ok(out)
}

#[derive(Default)]
struct ModuleOpts {
    name: Option<String>,
    skip: bool,
}

fn module_opts(attrs: &[Attribute]) -> syn::Result<ModuleOpts> {
    let mut out = ModuleOpts::default();

    for a in attrs {
        if !a.path().is_ident("module") {
            continue;
        }

        a.parse_nested_meta(|meta| {
            if meta.path.is_ident("name") {
                let value: LitStr = meta.value()?.parse()?;
                out.name = Some(value.value());
                Ok(())
            } else if meta.path.is_ident("skip") {
                // allow #[module(skip)] or #[module(skip = true/false)]
                if meta.input.peek(Token![=]) {
                    let v: LitBool = meta.value()?.parse()?;
                    out.skip = v.value();
                } else {
                    out.skip = true;
                }
                Ok(())
            } else {
                Err(meta.error("expected `name` or `skip`"))
            }
        })?;
    }

    Ok(out)
}

fn looks_like_config_type(ty: &Type) -> bool {
    // Heuristic: common "definitely not a module" types.
    // This is intentionally conservative: it only catches obvious stuff.
    let Type::Path(tp) = ty else { return false };
    let Some(seg) = tp.path.segments.last() else {
        return false;
    };
    let id = seg.ident.to_string();

    matches!(
        id.as_str(),
        "i8" | "i16"
            | "i32"
            | "i64"
            | "i128"
            | "isize"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "u128"
            | "usize"
            | "bool"
            | "f32"
            | "f64"
            | "String"
            | "PathBuf"
    )
}
